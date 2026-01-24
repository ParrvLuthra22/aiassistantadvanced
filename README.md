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

### Agents

| Agent | Responsibility |
|-------|---------------|
| **VoiceAgent** | Speech recognition (STT) and synthesis (TTS) |
| **IntentAgent** | Natural language understanding and intent classification |
| **SystemAgent** | macOS system commands and integrations |
| **MemoryAgent** | Conversation history and persistent memory |

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

- [ ] Wake word detection (Porcupine)
- [ ] Local Whisper STT
- [ ] HUD UI overlay
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
