import openwakeword
from openwakeword.model import Model
import pyaudio
import numpy as np

model = Model()

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

print("Listening for wake word...")
print("Try saying: hey jarvis")

while True:
    audio_data = stream.read(1280, exception_on_overflow=False)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    prediction = model.predict(audio_array)

    for wakeword, score in prediction.items():
        if score > 0.5:
            print("Wake word detected:", wakeword, score)
