"""
Step 3: split into two threads.

  Thread 1 (stt_worker):     mic -> Vosk -> complete sentence -> queue
  Thread 2 (command_worker): queue -> parse -> validate -> serial write

Why two threads here specifically: execute_rotate() blocks for ~0.3s
writing to and reading back from the Arduino. If that happened on the
same thread that's capturing audio, you'd lose whatever was said during
that window. Splitting them means the mic is never not-listening.

No TTS yet -- console output only, so this stage is easy to read and
debug. Confirmation/TTS can go back in once this is solid.
"""

import re
import sys
import json
import time
import queue
import threading
import argparse

import sounddevice as sd
import vosk
import serial
import serial.tools.list_ports

SAMPLE_RATE = 16000
BLOCKSIZE = 4000
SENTENCE_SILENCE = 1.5      # seconds of real silence before a sentence is "done"
MIN_CONFIDENCE = 0.5        # average word confidence below this -> reject and ask to repeat

DIGIT_WORDS = {
    "zero": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

COMMAND_GRAMMAR = '''
[
"forward", "back", "backward", "left", "right", "move", "turn", "rotate",
"zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
"centimeter", "centimeters", "meter", "meters", "metre", "metres",
"degree", "degrees", "yes", "no", "confirm", "cancel", "stop"
]
'''

audio_queue = queue.Queue()          # raw audio frames: mic callback -> stt_worker
command_queue = queue.Queue()        # recognized sentences: stt_worker -> command_worker
stop_event = threading.Event()

arduino_serial = None


# -----------------------------
# ARDUINO SERIAL
# -----------------------------
def find_and_connect_arduino():
    global arduino_serial
    ports = serial.tools.list_ports.comports()
    target_port = None
    for port in ports:
        if "Arduino" in port.description or "Mega" in port.description or port.vid == 0x2341:
            target_port = port.device
            break
        if "ttyACM" in port.device or "ttyUSB" in port.device:
            target_port = port.device

    if not target_port:
        print("ERROR: Could not find Arduino Mega serial port.")
        return False

    print(f"Connecting to Arduino on {target_port}...")
    try:
        arduino_serial = serial.Serial(target_port, 115200, timeout=3)
        time.sleep(2.5)
        if arduino_serial.in_waiting > 0:
            print(f"Arduino says: {arduino_serial.readline().decode('utf-8', errors='ignore').strip()}")
        print("Successfully connected to Arduino Mega!")
        return True
    except serial.SerialException as e:
        print(f"Serial Connection Error: {e}")
        return False


def execute_rotate(cmd):
    """Only function that touches arduino_serial -- confined to command_worker's thread."""
    global arduino_serial
    if not arduino_serial or not arduino_serial.is_open:
        print("[DEBUG] Serial connection unavailable.")
        return False

    target_angle = cmd["value"]  # absolute angle, 0-180
    clamped = max(0, min(180, target_angle))
    if clamped != target_angle:
        print(f"[DEBUG] Requested angle {target_angle} clamped to {clamped}")

    try:
        arduino_serial.reset_input_buffer()
        arduino_serial.write(f"{clamped}\n".encode("utf-8"))
        print(f"[DEBUG] Sent target angle: {clamped}")
        time.sleep(0.3)  # this is exactly the block that must not stall audio capture
        if arduino_serial.in_waiting > 0:
            print(f"[DEBUG] Arduino response: {arduino_serial.read_all().decode('utf-8', errors='ignore').strip()}")
        return True
    except serial.SerialException as e:
        print(f"[DEBUG] Failed to send command over serial: {e}")
        return False


# -----------------------------
# PARSING / VALIDATION
# -----------------------------
def extract_digit_number(text):
    digits = []
    started = False
    for word in text.lower().split():
        if word in DIGIT_WORDS:
            digits.append(str(DIGIT_WORDS[word]))
            started = True
        elif started:
            break
    return int("".join(digits)) if digits else None


def parse_command(text):
    text = text.lower()

    if "stop" in text or "cancel" in text:
        return {"action": "stop"}

    # Rotate is now an absolute target angle (0-180), not a left/right
    # offset from center -- maps directly onto the servo's own range and
    # the Arduino's parseInt() protocol, so no direction word is needed.
    if "rotate" in text or "turn" in text or "degree" in text:
        value = extract_digit_number(text)
        if value is None:
            return {"error": "number_not_understood"}
        if value > 180:
            return {"error": "angle_too_large"}
        return {"action": "rotate", "value": value, "unit": "degree"}

    action = direction = None
    if "forward" in text or "front" in text:
        action, direction = "move", "forward"
    elif "back" in text:
        action, direction = "move", "backward"
    elif "left" in text:
        action, direction = "move", "left"
    elif "right" in text:
        action, direction = "move", "right"

    if not action or not direction:
        return None

    value = extract_digit_number(text)
    if value is None:
        return {"error": "number_not_understood"}

    value_cm = value * 100 if ("meter" in text or "metre" in text) else value
    if value_cm > 100:
        return {"error": "distance_too_large"}
    return {"action": "move", "direction": direction, "value": value_cm, "unit": "centimeter"}


# -----------------------------
# THREAD 1: continuous listening
# -----------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[WARN] audio status: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))


def stt_worker(model_path, device):
    print(f"[STT] Loading Vosk model from '{model_path}'...")
    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE, COMMAND_GRAMMAR)
    recognizer.SetWords(True)

    sentence_parts, all_words = [], []
    last_partial = ""
    last_activity_time = time.time()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, dtype="int16",
        channels=1, device=device, callback=audio_callback,
    ):
        print("[STT] Listening (thread running independently of command processing)\n")
        while not stop_event.is_set():
            now = time.time()
            try:
                data = audio_queue.get(timeout=0.15)
            except queue.Empty:
                data = None

            if data is not None:
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        sentence_parts.append(text)
                        all_words.extend(result.get("result", []))
                        last_activity_time = now
                        print(f"  [STT] ...heard: \"{text}\"")
                    last_partial = ""
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    if partial:
                        last_activity_time = now
                    if partial != last_partial:
                        last_partial = partial
                        print(f"\r[STT] partial: {partial}" + " " * 15, end="", flush=True)

            if sentence_parts and (now - last_activity_time) > SENTENCE_SILENCE:
                full_text = " ".join(sentence_parts)
                avg_conf = (
                    sum(w["conf"] for w in all_words) / len(all_words)
                    if all_words else 0.0
                )
                print(f"\n[STT] SENTENCE COMPLETE: \"{full_text}\" (avg conf {avg_conf:.2f})")
                command_queue.put({"text": full_text, "confidence": avg_conf})
                sentence_parts, all_words = [], []
                last_partial = ""


# -----------------------------
# THREAD 2: parse, validate, act
# -----------------------------
def command_worker():
    while not stop_event.is_set():
        try:
            item = command_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        text, confidence = item["text"], item["confidence"]

        if confidence < MIN_CONFIDENCE:
            print(f"[CMD] Low confidence ({confidence:.2f}) on \"{text}\" -- ignoring, please repeat.")
            continue

        cmd = parse_command(text)
        if cmd is None:
            print(f"[CMD] Could not parse a command from: \"{text}\"")
            continue

        if "error" in cmd:
            print(f"[CMD] Validation error: {cmd['error']} (from \"{text}\")")
            continue

        if cmd["action"] == "stop":
            print("[CMD] STOP received.")
            continue

        print(f"[CMD] Parsed + validated: {cmd}")

        if cmd["action"] == "rotate":
            success = execute_rotate(cmd)
            print(f"[CMD] Serial send {'succeeded' if success else 'FAILED'}")
        elif cmd["action"] == "move":
            print("[CMD] Move is not wired to hardware yet -- parsed but not sent.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model")
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    if not find_and_connect_arduino():
        print("Initialization aborted: Arduino device not found.")
        return

    t_stt = threading.Thread(target=stt_worker, args=(args.model, args.device), daemon=True)
    t_cmd = threading.Thread(target=command_worker, daemon=True)
    t_stt.start()
    t_cmd.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
        t_stt.join(timeout=2)
        t_cmd.join(timeout=2)
        if arduino_serial and arduino_serial.is_open:
            arduino_serial.close()


if __name__ == "__main__":
    main()
