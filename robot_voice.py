import openwakeword
from openwakeword.model import Model

import pyaudio
import numpy as np

import queue
import sounddevice as sd
import vosk
import json
import time

# -----------------------------
# WAKE WORD SETUP
# -----------------------------

wake_model = Model()

audio = pyaudio.PyAudio()

wake_stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

# -----------------------------
# VOSK SETUP
# -----------------------------

model = vosk.Model("model")

grammar = '''
[
  "move forward one centimeter",
  "move forward two centimeter",
  "move forward three centimeter",
  "move forward four centimeter",
  "move forward five centimeter",
  "move forward ten centimeter",

  "move backward one centimeter",
  "move backward two centimeter",
  "move backward three centimeter",
  "move backward four centimeter",
  "move backward five centimeter",
  "move backward ten centimeter",

  "move left one centimeter",
  "move left two centimeter",
  "move left three centimeter",
  "move left four centimeter",
  "move left five centimeter",
  "move left ten centimeter",

  "move right one centimeter",
  "move right two centimeter",
  "move right three centimeter",
  "move right four centimeter",
  "move right five centimeter",
  "move right ten centimeter",

  "stop",
  "[unk]"
]
'''

samplerate = 16000

recognizer = vosk.KaldiRecognizer(
    model,
    samplerate,
    grammar
)

q = queue.Queue()

def callback(indata, frames, time_info, status):
    q.put(bytes(indata))

# -----------------------------
# MAIN LOOP
# -----------------------------

print("Waiting for wake word: Alexa")

while True:

    # Wake-word listening
    audio_data = wake_stream.read(
        1280,
        exception_on_overflow=False
    )

    audio_array = np.frombuffer(
        audio_data,
        dtype=np.int16
    )

    prediction = wake_model.predict(audio_array)

    wake_detected = False

    for wakeword, score in prediction.items():

        if wakeword == "alexa" and score > 0.75:
            print("\nWake word detected!")
            wake_detected = True
            break

    # If wake detected -> command mode
    if wake_detected:

        print("Listening for command...")

        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback
        ):

            start_time = time.time()

            while True:

                # timeout after 6 sec
                if time.time() - start_time > 6:
                    print("Command timeout")
                    break

                data = q.get()

                if recognizer.AcceptWaveform(data):

                    result = json.loads(
                        recognizer.Result()
                    )

                    text = result.get("text", "")

                    if text != "":
                        print("\nCOMMAND:", text)
                        break

        print("\nWaiting for wake word: Alexa")
