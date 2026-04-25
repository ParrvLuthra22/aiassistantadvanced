# FRIDAY Open-Source Stack (M3, 8GB RAM)

This guide is tuned for local-first performance on Apple Silicon with 8GB RAM.

## 1) Core local runtime

```bash
brew install ollama
ollama serve
```

## 2) Recommended local models

Use one at a time on 8GB RAM for best responsiveness.

### Intent + general reasoning

```bash
ollama pull qwen2.5:7b-instruct
```

### Coding help

```bash
ollama pull qwen2.5-coder:7b
```

### Fast lightweight fallback

```bash
ollama pull llama3.2:3b
```

### Optional lightweight vision captioning (non-OCR)

```bash
ollama pull moondream
```

## 3) FRIDAY settings

`config/settings.yaml` is already configured for local-first:

- `intent.provider: "ollama"`
- `vision.use_gemini: false`
- `vision.local_ocr_enabled: true`
- `security.face_auth.enabled: true`

## 4) Local face enrollment

When camera permissions are granted, enroll/update your face profile:

```bash
python3 scripts/enroll_face.py
```

If macOS blocked camera access for the current terminal/app:

```bash
tccutil reset Camera
```

Then re-run `python3 scripts/enroll_face.py`.

## 5) Voice stack

- Wake word: `friday`
- TTS: `kokoro` with `af_heart` (American female profile)
- Owner addressing: enabled via `voice.address_user_as_sir: true`

## 6) Performance notes for 8GB RAM

- Keep only one 7B model loaded at a time.
- Prefer quantized GGUF models via Ollama defaults.
- Close heavy apps while running local inference.
- Use OCR for text extraction; avoid large multimodal models for continuous screen parsing.
