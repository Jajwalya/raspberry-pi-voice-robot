import queue
import sounddevice as sd
import vosk
import sys
import json

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

model = vosk.Model("model")

samplerate = 16000
recognizer = vosk.KaldiRecognizer(model, samplerate)

print("Speak now. Press Ctrl+C to stop.")

with sd.RawInputStream(
    samplerate=samplerate,
    blocksize=8000,
    dtype="int16",
    channels=1,
    callback=callback
):
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print("You said:", text)
