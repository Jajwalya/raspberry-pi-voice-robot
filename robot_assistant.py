import os
import re
import time
import json
import queue

import numpy as np
import pyaudio
import sounddevice as sd
import vosk
from openwakeword.model import Model


# -----------------------------
# SPEAKER / TTS
# -----------------------------

def speak(text):
    print("ROBOT:", text)
    safe_text = text.replace('"', '')
    os.system(f'espeak-ng "{safe_text}"')


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
    words = words.lower().split()
    total = 0
    current = 0

    for word in words:
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
    text = text.lower()
    text = text.replace("-", " ")

    if "stop" in text or "cancel" in text:
        return {"action": "stop"}

    action = None
    direction = None

    if "rotate" in text or "turn" in text:
        action = "rotate"
        if "left" in text:
            direction = "left"
        elif "right" in text:
            direction = "right"
    elif "move" in text or "go" in text:
        action = "move"
        if "forward" in text or "front" in text:
            direction = "forward"
        elif "backward" in text or "back" in text:
            direction = "backward"
        elif "left" in text:
            direction = "left"
        elif "right" in text:
            direction = "right"

    if not action or not direction:
        return None

    digit_match = re.search(r"\b(\d+)\b", text)

    if digit_match:
        value = int(digit_match.group(1))
    else:
        number_words = []
        for word in text.split():
            if word in NUMBERS:
                number_words.append(word)

        value = text_number_to_int(" ".join(number_words))

    if value is None:
        return None

    if "meter" in text or "metre" in text:
        unit = "meter"
        value_cm = value * 100
    elif "millimeter" in text or "millimetre" in text or "mm" in text:
        unit = "millimeter"
        value_cm = value / 10
    elif "degree" in text or "degrees" in text:
        unit = "degree"
        value_cm = None
    else:
        unit = "centimeter"
        value_cm = value

    if action == "move":
        if value_cm > 100:
            return {
                "error": "distance_too_large",
                "max": 100
            }

        return {
            "action": "move",
            "direction": direction,
            "value": value_cm,
            "unit": "centimeter"
        }

    if action == "rotate":
        if value > 360:
            return {
                "error": "angle_too_large",
                "max": 360
            }

        return {
            "action": "rotate",
            "direction": direction,
            "value": value,
            "unit": "degree"
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
  "move forward one centimeter",
  "move forward two centimeter",
  "move forward three centimeter",
  "move forward four centimeter",
  "move forward five centimeter",
  "move forward six centimeter",
  "move forward seven centimeter",
  "move forward eight centimeter",
  "move forward nine centimeter",
  "move forward ten centimeter",
  "move forward twenty centimeter",
  "move forward thirty centimeter",
  "move forward fifty centimeter",
  "move forward one meter",

  "move backward one centimeter",
  "move backward two centimeter",
  "move backward three centimeter",
  "move backward four centimeter",
  "move backward five centimeter",
  "move backward six centimeter",
  "move backward seven centimeter",
  "move backward eight centimeter",
  "move backward nine centimeter",
  "move backward ten centimeter",
  "move backward twenty centimeter",
  "move backward thirty centimeter",
  "move backward fifty centimeter",
  "move backward one meter",

  "move left one centimeter",
  "move left two centimeter",
  "move left three centimeter",
  "move left four centimeter",
  "move left five centimeter",
  "move left six centimeter",
  "move left seven centimeter",
  "move left eight centimeter",
  "move left nine centimeter",
  "move left ten centimeter",
  "move left twenty centimeter",
  "move left thirty centimeter",
  "move left fifty centimeter",
  "move left one meter",

  "move right one centimeter",
  "move right two centimeter",
  "move right three centimeter",
  "move right four centimeter",
  "move right five centimeter",
  "move right six centimeter",
  "move right seven centimeter",
  "move right eight centimeter",
  "move right nine centimeter",
  "move right ten centimeter",
  "move right twenty centimeter",
  "move right thirty centimeter",
  "move right fifty centimeter",
  "move right one meter",

  "go forward ten centimeter",
  "go back ten centimeter",
  "go left ten centimeter",
  "go right ten centimeter",

  "rotate left ninety degree",
  "rotate right ninety degree",
  "turn left ninety degree",
  "turn right ninety degree",
  "rotate left forty five degree",
  "rotate right forty five degree",

  "yes",
  "no",
  "confirm",
  "cancel",
  "stop"
]
'''


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))


def listen_once(timeout=8):
    recognizer = vosk.KaldiRecognizer(
        vosk_model,
        samplerate,
        COMMAND_GRAMMAR
    )

    while not audio_queue.empty():
        audio_queue.get()

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                return ""

            data = audio_queue.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print("HEARD:", text)
                    return text


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
    frames_per_buffer=1280
)


def wait_for_wake_word():
    print("Waiting for wake word: Alexa")

    while True:
        audio_data = wake_stream.read(
            1280,
            exception_on_overflow=False
        )

        audio_array = np.frombuffer(
            audio_data,
            dtype=np.int16
        )

        prediction = wake_model.predict(audio_array)

        for wakeword, score in prediction.items():
            if wakeword == "alexa" and score > 0.75:
                print("Wake word detected:", wakeword, score)
                return


# -----------------------------
# MAIN ASSISTANT LOOP
# -----------------------------

def main():
    speak("Robot assistant started")

    while True:
        wait_for_wake_word()

        speak("Hello, what can I do?")

        command_text = listen_once(timeout=8)

        if not command_text:
            speak("I did not hear a command")
            continue

        cmd = parse_command(command_text)

        if cmd is None:
            speak("Sorry, I did not understand the command")
            continue

        if "error" in cmd:
            if cmd["error"] == "distance_too_large":
                speak("Distance is too large. Maximum allowed is 100 centimeters")
            elif cmd["error"] == "angle_too_large":
                speak("Angle is too large. Maximum allowed is 360 degrees")
            continue

        if cmd["action"] == "stop":
            speak("Stopping")
            print("COMMAND CONFIRMED: STOP")
            continue

        sentence = command_to_sentence(cmd)

        speak(f"Do you want me to {sentence}?")

        answer = listen_once(timeout=6)

        if answer in ["yes", "confirm"]:
            speak(f"Okay, {sentence}")
            print("COMMAND CONFIRMED:", cmd)

            # TODO: later call motor function here
            # execute_robot_command(cmd)

        elif answer in ["no", "cancel", "stop"]:
            speak("Command cancelled")
            print("COMMAND CANCELLED")

        else:
            speak("I did not get confirmation")
            print("NO CONFIRMATION")


if __name__ == "__main__":
    main()
