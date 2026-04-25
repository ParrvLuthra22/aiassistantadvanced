# FRIDAY Personal Assistant

FRIDAY is a modular, event-driven macOS assistant with voice control, local intent parsing, screen OCR, automation tooling, and an HUD.

This project is designed for local-first usage on Apple Silicon with optional cloud integrations.

## What FRIDAY Can Do

### Voice + Conversation
- Wake word listening (`friday`)
- Speech-to-text pipeline (Vosk + Whisper.cpp flow)
- Text-to-speech with Kokoro (`af_heart`) and fallback to `pyttsx3`
- Polite owner addressing (`Sir`) in spoken responses

### Identity + Access Control
- Startup face verification gate before command handling
- Local face profile enrollment and verification
- Configurable identity threshold and camera source
- Startup spoken flow:
  - `friday starting verifying for user`
  - `Verification Successful, Welcome Parrv Luthra`
  - `Hello Sir what are we working on today`

### Vision / Screen Understanding (Local)
- Full screen capture (`/tmp/friday_screen.png`)
- Local OCR via Tesseract (no Gemini required)
- Screen reading intents such as:
  - “what’s on my screen”
  - “read my screen”
  - “read that”
  - “where is X on screen”

### macOS Control
- App opening/switching
- Safari web opening/search flows
- Finder operations
- Calendar operations
- Spotify controls
- System controls (volume, brightness CLI, Wi-Fi toggle, dark mode, battery query)

### Web + Research
- Web search agent with Tavily + LLM summarization
- Supports OpenRouter/Grok or Gemini for query extraction and summarization
- If Tavily is not configured, FRIDAY can still answer in LLM-only mode (no source links)
- HUD updates with current command/response and event transcript

### Image Generation
- Local SDXL-Turbo image generation agent (Apple Silicon path)
- HUD image event rendering for generated outputs

### Multi-step Reasoning
- LangGraph-based reasoning engine scaffold for complex multi-tool requests
- Plan -> execute -> verify -> respond workflow

## Architecture Overview

- `orchestrator/brain.py`: lifecycle + routing + startup orchestration
- `bus/event_bus.py`: pub/sub backbone
- `agents/*`: domain-specific capability modules
- `schemas/events.py`: typed event contracts
- `ui/hud_overlay.py`: floating HUD
- `config/settings.yaml`: runtime configuration

## Quick Start

## 1) Create venv and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional API keys:
- `OPENROUTER_API_KEY` for Grok/OpenRouter-based extraction and summarization
- `TAVILY_API_KEY` for source-backed live web retrieval

## 2) Install required macOS dependencies

```bash
brew install portaudio tesseract brightness
```

If you use Whisper.cpp binary mode, ensure `voice.whisper.binary_path` points to a valid binary.

## 3) Ensure Kokoro model files exist

Expected files in repo root:
- `kokoro-v0_19.onnx`
- `voices.bin`

## 4) Run FRIDAY

```bash
./.venv/bin/python main.py
```

## Face Verification Setup (Required for locked startup)

FRIDAY is configured to enforce face verification at startup (`security.face_auth.enabled: true`).

## Step A: grant camera permission

On macOS:
- System Settings -> Privacy & Security -> Camera
- Enable access for whichever app runs Python:
  - Terminal / iTerm / VS Code (if using integrated terminal)

## Step B: enroll your face profile

```bash
./.venv/bin/python scripts/enroll_face.py --camera-id 0
```

If camera `0` fails, try:

```bash
./.venv/bin/python scripts/enroll_face.py --camera-id 1
```

Enrollment stores profile data in `data/face_auth`.

## Step C: verify startup phrase

After restart, FRIDAY should speak in order:
1. `friday starting verifying for user`
2. `Verification Successful, Welcome Parrv Luthra`
3. `Hello Sir what are we working on today`

If verification fails, commands remain blocked.

## Configuration You’ll Likely Edit

Primary file: `config/settings.yaml`

- `general.assistant_name`: currently `FRIDAY`
- `voice.wake_word`: currently `friday`
- `voice.tts.voice_id`: currently `af_heart`
- `voice.address_user_as_sir`: true
- `security.face_auth.owner_name`: currently `parrv luthra`
- `security.face_auth.enabled`: true/false
- `security.face_auth.camera_id`: camera index
- `vision.local_ocr_enabled`: true
- `intent.provider`: `ollama` (recommended local) or `pattern`
- `web_search.llm_provider`: `auto`, `openrouter`, `gemini`, or `local`
- `web_search.openrouter.model`: set to your preferred Grok/OpenRouter model

## Manual Test Plan

Use these spoken commands (or keyboard input path, depending on your run mode):

1. Identity and greeting
- Start FRIDAY and confirm verification greeting is spoken.

2. Voice response baseline
- “Friday, what time is it?”

3. Screen OCR
- Put visible text on screen, then say:
  - “What’s on my screen?”
  - “Read all text on my screen”

4. App control
- “Open Safari”
- “Search for Python context managers”

5. System control
- “Set volume to 40”
- “What’s my battery?”

6. Finder/utility checks
- “Open downloads”

7. Image generation
- “Generate image of a futuristic assistant dashboard”

8. Search acknowledgement
- “Search for best Python async tutorial”
- FRIDAY should first say: `Thats a great idea sir`

## Troubleshooting

### `Face enrollment failed: no clear face detected`
- Improve lighting and face framing.
- Close apps occupying camera (Zoom/Meet/FaceTime).
- Recheck macOS camera permissions.

### `camera access has been denied`
- macOS permission issue, not model issue.
- Grant camera access to the app launching Python.

### No speech audio
- Check selected audio output device.
- Confirm Kokoro files exist.
- FRIDAY falls back to `pyttsx3` on Kokoro failure.

### OCR not working
- Confirm `tesseract` is installed (`brew install tesseract`).
- Ensure `pytesseract` is installed in the same venv.

## Project Status

This is an actively evolving assistant stack. Some advanced flows are experimental and improving in each iteration.

For your next pass, run the manual test plan and report exactly which command failed + what FRIDAY said/logged. That gives the fastest fix cycle.
