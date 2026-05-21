# Raspberry Pi Voice Robot Assistant

Offline voice assistant for a Raspberry Pi based robot.

## Features

- Wake word detection using openWakeWord
- Speech-to-text using Vosk
- Voice acknowledgement using espeak-ng
- Command confirmation before robot movement
- Supports commands like:
  - move forward ten centimeter
  - move left ten centimeter
  - rotate right ninety degree

## Current Status

Phase 1: Voice command assistant working.

## Run

```bash
cd ~/speech-to-text
source venv/bin/activate
python robot_assistant.py
