import os
import re
import time
import json
import queue
import subprocess

import numpy as np
import pyaudio
import sounddevice as sd
import vosk
from openwakeword.model import Model


# -----------------------------
# PIPER TTS
# -----------------------------

PIPER_BIN = "./piper/piper"
PIPER_MODEL = "voices/en_US-lessac-medium.onnx"
RESPONSE_WAV = "response.wav"


def speak(text):
    print("ROBOT:", text)

    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", RESPONSE_WAV],
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    subprocess.run(
        ["aplay", RESPONSE_WAV],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


# -----------------------------
# WORD TO NUMBER
# -----------------------------

NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}


def text_number_to_int(words):
    total = 0
    current = 0

    for word in words.lower().split():
        if word in NUMBERS:
            value = NUMBERS[word]
            if value == 100:
                current = max(1, current) * 100
            else:
                current += value

    total += current
    return total if total > 0 else None


# -----------------------------
# COMMAND PARSER
# -----------------------------

def parse_command(text):
    text = text.lower().replace("-", " ")

    if "stop" in text or "cancel" in text:
        return {"action": "stop"}

    digit_match = re.search(r"\b(\d+)\b", text)

    if digit_match:
        value = int(digit_match.group(1))
    else:
        number_words = [word for word in text.split() if word in NUMBERS]
        value = text_number_to_int(" ".join(number_words))

    if value is None:
        value = 10

    action = None
    direction = None

    if "forward" in text or "front" in text:
        action = "move"
        direction = "forward"

    elif "back" in text or "backward" in text:
        action = "move"
        direction = "backward"

    elif "left" in text:
        direction = "left"
        if "turn" in text or "rotate" in text or "degree" in text:
            action = "rotate"
        else:
            action = "move"

    elif "right" in text:
        direction = "right"
        if "turn" in text or "rotate" in text or "degree" in text:
            action = "rotate"
        else:
            action = "move"

    if not action or not direction:
        return None

    if action == "move":
        value_cm = value

        if "meter" in text or "metre" in text:
            value_cm = value * 100

        if value_cm > 100:
            return {"error": "distance_too_large"}

        return {
            "action": "move",
            "direction": direction,
            "value": value_cm,
            "unit": "centimeter",
        }

    if action == "rotate":
        if value > 360:
            return {"error": "angle_too_large"}

        return {
            "action": "rotate",
            "direction": direction,
            "value": value,
            "unit": "degree",
        }

    return None


def command_to_sentence(cmd):
    if cmd["action"] == "move":
        return f"move {cmd['direction']} {cmd['value']} centimeters"

    if cmd["action"] == "rotate":
        return f"rotate {cmd['direction']} {cmd['value']} degrees"

    if cmd["action"] == "stop":
        return "stop"

    return "unknown command"


# -----------------------------
# VOSK LISTENER
# -----------------------------

samplerate = 16000
audio_queue = queue.Queue()
vosk_model = vosk.Model("model")

COMMAND_GRAMMAR = '''
[
  "forward",
  "back",
  "backward",
  "left",
  "right",

  "move",
  "turn",
  "rotate",

  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "twenty",
  "thirty",
  "forty",
  "fifty",
  "ninety",

  "centimeter",
  "centimeters",
  "meter",
  "degree",
  "degrees",

  "yes",
  "no",
  "confirm",
  "cancel",
  "stop"
]
'''


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))


def listen_once(timeout=6):
    recognizer = vosk.KaldiRecognizer(
        vosk_model,
        samplerate,
        COMMAND_GRAMMAR,
    )

    while not audio_queue.empty():
        audio_queue.get()

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=4000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        start_time = time.time()
        last_partial = ""

        while True:
            if time.time() - start_time > timeout:
                if last_partial:
                    print("TIMEOUT USING PARTIAL:", last_partial)
                    return last_partial
                return ""

            data = audio_queue.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")

                if text:
                    print("FINAL:", text)
                    return text

            else:
                partial = json.loads(
                    recognizer.PartialResult()
                ).get("partial", "")

                if partial and partial != last_partial:
                    last_partial = partial
                    print("PARTIAL:", partial)


# -----------------------------
# WAKE WORD
# -----------------------------

wake_model = Model()
pa = pyaudio.PyAudio()

wake_stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280,
)


def wait_for_wake_word():
    print("Waiting for wake word: Alexa")

    while True:
        audio_data = wake_stream.read(
            1280,
            exception_on_overflow=False,
        )

        audio_array = np.frombuffer(
            audio_data,
            dtype=np.int16,
        )

        prediction = wake_model.predict(audio_array)

        for wakeword, score in prediction.items():
            if wakeword == "alexa" and score > 0.85:
                print("Wake word detected:", wakeword, score)
                return


# -----------------------------
# MAIN LOOP
# -----------------------------

def main():
    speak("Robot assistant started")

    while True:
        wait_for_wake_word()

        speak("Ready")

        command_text = listen_once(timeout=6)

        if not command_text:
            speak("I did not hear you")
            continue

        cmd = parse_command(command_text)

        if cmd is None:
            speak("Command not understood")
            continue

        if "error" in cmd:
            if cmd["error"] == "distance_too_large":
                speak("Distance too large")
            elif cmd["error"] == "angle_too_large":
                speak("Angle too large")
            continue

        if cmd["action"] == "stop":
            speak("Stopping")
            print("COMMAND CONFIRMED: STOP")
            continue

        sentence = command_to_sentence(cmd)

        speak(f"Confirm {sentence}")

        answer = listen_once(timeout=5)

        if answer in ["yes", "confirm"]:
            speak("Okay")
            print("COMMAND CONFIRMED:", cmd)

            # TODO: connect motor code here
            # execute_robot_command(cmd)

        elif answer in ["no", "cancel", "stop"]:
            speak("Cancelled")
            print("COMMAND CANCELLED")

        else:
            speak("No confirmation")
            print("NO CONFIRMATION")


if __name__ == "__main__":
    main()
