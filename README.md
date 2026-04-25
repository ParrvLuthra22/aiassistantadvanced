# JARVIS Virtual Assistant

A modular, event-driven virtual assistant for macOS, inspired by JARVIS from Iron Man.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Brain                                    │
│                    (Orchestrator)                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Event Bus                                 │ │
│  │              (Pub/Sub Communication)                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         ▲              ▲              ▲              ▲           │
│         │              │              │              │           │
│    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐       │
│    │  Voice  │   │ Intent  │   │ System  │   │ Memory  │       │
│    │  Agent  │   │  Agent  │   │  Agent  │   │  Agent  │       │
│    └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

- **Brain (Orchestrator)**: Central coordinator managing agent lifecycle and health
- **Event Bus**: Pub/sub system for decoupled agent communication
- **Agents**: Specialized modules handling specific domains
- **Health API**: FastAPI endpoint for observability (GET `/health`)

### Agents

| Agent | Responsibility |
|-------|---------------|
| **VoiceAgent** | Speech recognition (STT) and synthesis (TTS) |
| **IntentAgent** | Natural language understanding and intent classification |
| **SystemAgent** | macOS system commands and integrations |
| **MemoryAgent** | Conversation history and persistent memory |
| **PluginAgent** | Loads Python plugins from `plugins/` and routes intents |

## Project Structure

```
jarvis/
├── main.py                 # Entry point
├── orchestrator/
│   └── brain.py           # Central coordinator
├── agents/
│   ├── base_agent.py      # Abstract base class
│   ├── voice_agent.py     # Speech I/O
│   ├── intent_agent.py    # NLU processing
│   ├── system_agent.py    # macOS integration
│   └── memory_agent.py    # Context management
├── bus/
│   └── event_bus.py       # Pub/sub event system
├── schemas/
│   └── events.py          # Event type definitions
├── config/
│   └── settings.yaml      # Configuration
├── utils/
│   └── logger.py          # Logging utilities
└── requirements.txt       # Python dependencies
```

## Quick Start

### Prerequisites

- macOS (tested on Monterey and later)
- Python 3.10+
- Gemini API key (for intent recognition)
- PortAudio (for microphone access)

### Installation

```bash
# Clone the repository
cd aiassistanttrying

# Install PortAudio (macOS)
brew install portaudio

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install voice dependencies (FREE, offline)
pip install vosk pyttsx3 pyaudio SpeechRecognition

# Set up environment variables
export GEMINI_API_KEY="your-api-key-here"

# Run the assistant
python3 main.py
```

## Observability

JARVIS exposes a health endpoint for basic observability:

- **Endpoint:** `GET http://localhost:8080/health`
- **Response shape:**

```json
{
    "status": "ok",
    "agents": {
        "VoiceAgent": "healthy",
        "IntentAgent": "healthy",
        "SystemAgent": "healthy"
    }
}
```

The server starts automatically when the Brain starts and shuts down during shutdown.

## Plugins

Place Python plugins in `plugins/` and they are auto-loaded at startup.

Each plugin must define:

```python
PLUGIN_NAME = "MyPlugin"
TRIGGERS = ["my intent", "another intent"]

async def handle(event):
        return "Response text"
```

Example plugin: `plugins/calculator_plugin.py`.

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push and PRs to `main`:

- Python 3.10
- Installs `requirements.txt`
- Runs `pytest tests/ --cov=. --cov-config=.coveragerc --cov-fail-under=40`

### Voice Setup (Optional but Recommended)

For full voice capabilities, install:

#### 1. Vosk Wake Word Detection (FREE, offline)

```bash
# Download Vosk model
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

#### 2. Whisper.cpp Transcription (FREE, offline, high accuracy)

```bash
# Build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make

# Download model
./models/download-ggml-model.sh base.en

# Install binary
sudo cp main /usr/local/bin/whisper-cpp
cp models/ggml-base.en.bin ../models/
cd ..
```

### Configuration

Edit `config/settings.yaml` to customize:

- Wake word (`voice.wake_word`)
- Vosk model path (`voice.vosk.model_path`)
- Whisper binary/model paths (`voice.whisper.*`)
- TTS voice and rate (`voice.synthesis.*`)
- Silence detection thresholds

## Voice Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VoiceAgent                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Microphone  │──▶│ Vosk Wake   │──▶│ whisper.cpp │           │
│  │  Stream     │   │ Word Detect │   │ Transcriber │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│         │                │                  │                   │
│         │    "jarvis"    │                  │                   │
│         │    detected    │                  ▼                   │
│         │                │         ┌─────────────┐              │
│         │                └────────▶│   Emit      │              │
│         │                          │VoiceInput   │              │
│         │                          │   Event     │              │
│         │                          └─────────────┘              │
│         │                                                       │
│         │         ┌─────────────┐   ┌─────────────┐            │
│         └────────▶│  pyttsx3    │◀──│ Subscribe   │            │
│                   │    TTS      │   │VoiceOutput  │            │
│                   └─────────────┘   │   Event     │            │
│                                     └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- 🎤 **Wake Word**: Vosk (FREE, runs locally)
- 🗣️ **Speech-to-Text**: whisper.cpp (FREE, high accuracy, offline)
- 🔊 **Text-to-Speech**: pyttsx3 (FREE, uses macOS native voices)
- 🧵 **Non-blocking**: Background threads for audio processing
- 📡 **Event-driven**: All communication via EventBus

## Usage Examples

Once running, speak to JARVIS:

- "Open Safari"
- "Search for Python tutorials"
- "What time is it?"
- "Set volume to 50"
- "What can you do?"

## Extending the System

### Adding a New Agent

1. Create a new agent in `agents/`:

```python
from agents.base_agent import BaseAgent, AgentCapability

class MyAgent(BaseAgent):
    @property
    def capabilities(self):
        return [AgentCapability(
            name="my_capability",
            description="What it does"
        )]
    
    async def _setup(self):
        self._subscribe(SomeEvent, self._handler)
    
    async def _teardown(self):
        pass
    
    async def _handler(self, event):
        # Handle the event
        pass
```

2. Register it in `orchestrator/brain.py`

### Adding New Events

Define new events in `schemas/events.py`:

```python
@dataclass(frozen=True)
class MyNewEvent(BaseEvent):
    my_field: str = ""
    source: str = field(default="my_agent")
```

## Future Roadmap

## Phase 1 RAG Memory (Implemented)

The assistant now includes a **Chroma-backed semantic memory pipeline** integrated with `MemoryAgent`.

### What it does

- Stores short-term and long-term semantic memory chunks in Chroma
- Uses metadata-aware retrieval with recency + salience scoring
- Provides diversified retrieval (MMR) and context assembly utilities
- Keeps existing SQLite memory behavior as-is

### Configuration

In `config/settings.yaml`:

- `memory.vector_store.enabled`
- `memory.vector_store.provider`
- `memory.vector_store.persist_directory`
- `memory.vector_store.collection_name`
- `memory.vector_store.embedding_model` (optional)

### Smoke test

Run the Phase 1 verification script:

```bash
python3 scripts/rag_phase1_smoke.py
```

Expected: it ingests sample memories and prints retrieved semantic matches.

## HUD Overlay (Implemented)

A live macOS HUD overlay is now integrated and starts automatically with the Brain.

### HUD features

- Borderless, always-on-top semi-transparent dark window
- Pulsing circular waveform while `VoiceAgent` is listening
- Last spoken command and latest assistant response
- Live EventBus log (last 5 events: timestamp + event type + source)
- Agent health row with status dots for:
    - `VoiceAgent`
    - `IntentAgent`
    - `SystemAgent`
    - `MemoryAgent`

### HUD configuration

In `config/settings.yaml` under `ui.hud`:

- `enabled`
- `width`, `height`, `x`, `y`
- `alpha`
- `background`

- [ ] Wake word detection (Porcupine)
- [ ] Local Whisper STT
- [ ] OpenCV integration
- [ ] Multi-agent workflows
- [ ] Plugin system
- [ ] Remote control API

## Design Principles

- **Event-Driven**: All communication via events
- **Modular**: Agents are independent and replaceable
- **Scalable**: Easy to add new agents and capabilities
- **SOLID**: Clean architecture following SOLID principles

## License

MIT License - feel free to use and modify.
