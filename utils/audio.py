"""
Audio utilities for the JARVIS Virtual Assistant.

This module provides helper functions for:
- Microphone access and audio recording
- Audio format conversion
- Silence/speech detection
- Voice Activity Detection (VAD)
- Audio buffer management

All audio operations are designed to be non-blocking where possible.
"""

import asyncio
import audioop
import io
import logging
import struct
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit audio
DEFAULT_CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION_MS = 1500


# =============================================================================
# Audio State
# =============================================================================

class AudioState(Enum):
    """State of the audio capture system."""
    IDLE = auto()
    LISTENING = auto()
    RECORDING = auto()
    PROCESSING = auto()
    ERROR = auto()


@dataclass
class AudioConfig:
    """Configuration for audio capture."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    sample_width: int = DEFAULT_SAMPLE_WIDTH
    chunk_size: int = DEFAULT_CHUNK_SIZE
    silence_threshold: int = SILENCE_THRESHOLD
    silence_duration_ms: int = SILENCE_DURATION_MS
    max_recording_seconds: float = 30.0
    min_recording_seconds: float = 0.5
    buffer_seconds: float = 0.5
    vad_aggressiveness: int = 2  # 0-3


@dataclass
class AudioChunk:
    """A chunk of audio data with metadata."""
    data: bytes
    timestamp: float = field(default_factory=time.time)
    rms: int = 0
    is_speech: bool = False
    
    def __post_init__(self):
        if self.rms == 0 and self.data:
            self.rms = calculate_rms(self.data)


# =============================================================================
# Audio Helper Functions
# =============================================================================

def calculate_rms(audio_data: bytes, sample_width: int = 2) -> int:
    """
    Calculate the Root Mean Square (RMS) of audio data.
    
    RMS is a measure of audio loudness/energy.
    
    Args:
        audio_data: Raw audio bytes
        sample_width: Bytes per sample (2 for 16-bit)
    
    Returns:
        RMS value (0-32768 for 16-bit audio)
    """
    if not audio_data:
        return 0
    try:
        return audioop.rms(audio_data, sample_width)
    except Exception:
        return 0


def is_silence(audio_data: bytes, threshold: int = SILENCE_THRESHOLD) -> bool:
    """
    Check if audio chunk is silence based on RMS threshold.
    
    Args:
        audio_data: Raw audio bytes
        threshold: RMS threshold below which is considered silence
    
    Returns:
        True if audio is below threshold (silence)
    """
    return calculate_rms(audio_data) < threshold


def normalize_audio(audio_data: bytes, sample_width: int = 2, target_rms: int = 8000) -> bytes:
    """
    Normalize audio volume to target RMS level.
    
    Args:
        audio_data: Raw audio bytes
        sample_width: Bytes per sample
        target_rms: Target RMS level
    
    Returns:
        Normalized audio bytes
    """
    if not audio_data:
        return audio_data
    
    current_rms = calculate_rms(audio_data, sample_width)
    if current_rms == 0:
        return audio_data
    
    try:
        factor = target_rms / current_rms
        # Clamp factor to prevent distortion
        factor = min(factor, 4.0)
        return audioop.mul(audio_data, sample_width, factor)
    except Exception:
        return audio_data


def convert_sample_rate(
    audio_data: bytes,
    from_rate: int,
    to_rate: int,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    """
    Convert audio sample rate.
    
    Args:
        audio_data: Raw audio bytes
        from_rate: Original sample rate
        to_rate: Target sample rate
        sample_width: Bytes per sample
        channels: Number of channels
    
    Returns:
        Resampled audio bytes
    """
    if from_rate == to_rate:
        return audio_data
    
    try:
        converted, _ = audioop.ratecv(
            audio_data,
            sample_width,
            channels,
            from_rate,
            to_rate,
            None,
        )
        return converted
    except Exception as e:
        logger.error(f"Failed to convert sample rate: {e}")
        return audio_data


def audio_to_wav_bytes(
    audio_data: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> bytes:
    """
    Convert raw audio bytes to WAV format in memory.
    
    Args:
        audio_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Bytes per sample
    
    Returns:
        WAV file bytes
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return buffer.getvalue()


def save_audio_to_file(
    audio_data: bytes,
    file_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> bool:
    """
    Save raw audio data to a WAV file.
    
    Args:
        audio_data: Raw PCM audio bytes
        file_path: Path to save the WAV file
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Bytes per sample
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        return True
    except Exception as e:
        logger.error(f"Failed to save audio to {file_path}: {e}")
        return False


def create_temp_wav_file(
    audio_data: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    prefix: str = "jarvis_audio_",
) -> Optional[str]:
    """
    Create a temporary WAV file from audio data.
    
    The caller is responsible for deleting the file.
    
    Args:
        audio_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Bytes per sample
        prefix: Prefix for temp file name
    
    Returns:
        Path to temp file, or None if failed
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            prefix=prefix,
            delete=False,
        )
        temp_path = temp_file.name
        temp_file.close()
        
        if save_audio_to_file(audio_data, temp_path, sample_rate, channels, sample_width):
            return temp_path
        return None
    except Exception as e:
        logger.error(f"Failed to create temp WAV file: {e}")
        return None


# =============================================================================
# Audio Buffer
# =============================================================================

class AudioBuffer:
    """
    Thread-safe circular audio buffer.
    
    Maintains a rolling buffer of audio chunks for:
    - Pre-wake-word audio capture
    - Continuous recording with backlog
    
    Usage:
        buffer = AudioBuffer(max_seconds=5.0)
        buffer.add_chunk(audio_data)
        all_audio = buffer.get_all()
    """
    
    def __init__(
        self,
        max_seconds: float = 5.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        channels: int = DEFAULT_CHANNELS,
    ):
        self._max_seconds = max_seconds
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self._channels = channels
        
        # Calculate max bytes
        bytes_per_second = sample_rate * sample_width * channels
        self._max_bytes = int(max_seconds * bytes_per_second)
        
        self._buffer: Deque[bytes] = deque()
        self._current_bytes = 0
        self._lock = threading.Lock()
    
    def add_chunk(self, audio_data: bytes) -> None:
        """Add audio chunk to buffer, removing old data if needed."""
        with self._lock:
            self._buffer.append(audio_data)
            self._current_bytes += len(audio_data)
            
            # Remove old chunks if over limit
            while self._current_bytes > self._max_bytes and self._buffer:
                removed = self._buffer.popleft()
                self._current_bytes -= len(removed)
    
    def get_all(self) -> bytes:
        """Get all audio data in buffer."""
        with self._lock:
            return b''.join(self._buffer)
    
    def get_last_seconds(self, seconds: float) -> bytes:
        """Get the last N seconds of audio."""
        bytes_per_second = self._sample_rate * self._sample_width * self._channels
        target_bytes = int(seconds * bytes_per_second)
        
        with self._lock:
            all_data = b''.join(self._buffer)
            if len(all_data) <= target_bytes:
                return all_data
            return all_data[-target_bytes:]
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._current_bytes = 0
    
    @property
    def duration_seconds(self) -> float:
        """Get current buffer duration in seconds."""
        bytes_per_second = self._sample_rate * self._sample_width * self._channels
        return self._current_bytes / bytes_per_second if bytes_per_second > 0 else 0


# =============================================================================
# Voice Activity Detection (VAD)
# =============================================================================

class SimpleVAD:
    """
    Simple Voice Activity Detection based on energy/RMS thresholds.
    
    This is a lightweight alternative to webrtcvad for basic speech detection.
    
    Usage:
        vad = SimpleVAD(threshold=500)
        for chunk in audio_stream:
            if vad.is_speech(chunk):
                process_speech(chunk)
    """
    
    def __init__(
        self,
        threshold: int = SILENCE_THRESHOLD,
        min_speech_frames: int = 3,
        min_silence_frames: int = 15,
    ):
        """
        Initialize VAD.
        
        Args:
            threshold: RMS threshold for speech detection
            min_speech_frames: Minimum consecutive speech frames to trigger
            min_silence_frames: Minimum consecutive silence frames to end
        """
        self._threshold = threshold
        self._min_speech_frames = min_speech_frames
        self._min_silence_frames = min_silence_frames
        
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
    
    def is_speech(self, audio_data: bytes) -> bool:
        """
        Check if audio chunk contains speech.
        
        Uses hysteresis to avoid rapid state changes.
        
        Args:
            audio_data: Raw audio bytes
        
        Returns:
            True if speech is detected
        """
        rms = calculate_rms(audio_data)
        
        if rms >= self._threshold:
            self._speech_frames += 1
            self._silence_frames = 0
            
            if self._speech_frames >= self._min_speech_frames:
                self._in_speech = True
        else:
            self._silence_frames += 1
            self._speech_frames = 0
            
            if self._silence_frames >= self._min_silence_frames:
                self._in_speech = False
        
        return self._in_speech
    
    def reset(self) -> None:
        """Reset VAD state."""
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
    
    @property
    def in_speech(self) -> bool:
        """Check if currently in speech segment."""
        return self._in_speech


# =============================================================================
# Microphone Capture
# =============================================================================

class MicrophoneStream:
    """
    Non-blocking microphone audio stream using PyAudio.
    
    Captures audio in a background thread and provides chunks via callback.
    
    Usage:
        def on_audio(chunk: bytes):
            process_chunk(chunk)
        
        mic = MicrophoneStream(callback=on_audio)
        mic.start()
        # ... later ...
        mic.stop()
    """
    
    def __init__(
        self,
        callback: Callable[[bytes], None],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        device_index: Optional[int] = None,
    ):
        """
        Initialize microphone stream.
        
        Args:
            callback: Function to call with each audio chunk
            sample_rate: Sample rate in Hz
            channels: Number of channels (1 = mono)
            chunk_size: Frames per buffer
            device_index: Specific input device index (None = default)
        """
        self._callback = callback
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device_index = device_index
        
        self._pyaudio = None
        self._stream = None
        self._running = False
        self._error: Optional[str] = None
    
    def start(self) -> bool:
        """
        Start capturing audio.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            import pyaudio
            
            self._pyaudio = pyaudio.PyAudio()
            
            # Check for available input devices
            info = self._pyaudio.get_default_input_device_info()
            logger.debug(f"Using input device: {info['name']}")
            
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=self._chunk_size,
                stream_callback=self._audio_callback,
            )
            
            self._running = True
            self._stream.start_stream()
            logger.info("Microphone stream started")
            return True
            
        except ImportError:
            self._error = "PyAudio not installed. Run: pip install pyaudio"
            logger.error(self._error)
            return False
        except OSError as e:
            self._error = f"Microphone access error: {e}"
            logger.error(self._error)
            return False
        except Exception as e:
            self._error = f"Failed to start microphone: {e}"
            logger.error(self._error)
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - called from audio thread."""
        import pyaudio
        
        if self._running and in_data:
            try:
                self._callback(in_data)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
        
        return (None, pyaudio.paContinue if self._running else pyaudio.paComplete)
    
    def stop(self) -> None:
        """Stop capturing audio."""
        self._running = False
        
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
            self._stream = None
        
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception as e:
                logger.debug(f"Error terminating PyAudio: {e}")
            self._pyaudio = None
        
        logger.info("Microphone stream stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running
    
    @property
    def error(self) -> Optional[str]:
        """Get last error message."""
        return self._error


# =============================================================================
# Audio Recorder
# =============================================================================

class AudioRecorder:
    """
    Records audio with automatic silence detection.
    
    Starts recording on speech detection and stops after silence timeout.
    
    Usage:
        recorder = AudioRecorder()
        recorder.start_recording()
        # ... audio chunks are added via add_chunk() ...
        if recorder.is_complete:
            audio = recorder.get_recording()
    """
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
    ):
        self._config = config or AudioConfig()
        self._vad = SimpleVAD(
            threshold=self._config.silence_threshold,
        )
        
        self._recording: List[bytes] = []
        self._is_recording = False
        self._speech_started = False
        self._silence_start: Optional[float] = None
        self._recording_start: Optional[float] = None
        self._lock = threading.Lock()
    
    def start_recording(self) -> None:
        """Start a new recording session."""
        with self._lock:
            self._recording.clear()
            self._is_recording = True
            self._speech_started = False
            self._silence_start = None
            self._recording_start = time.time()
            self._vad.reset()
        logger.debug("Recording started")
    
    def add_chunk(self, audio_data: bytes) -> None:
        """Add audio chunk to recording."""
        if not self._is_recording:
            return
        
        with self._lock:
            self._recording.append(audio_data)
            
            # Check for speech
            is_speech = self._vad.is_speech(audio_data)
            
            if is_speech:
                self._speech_started = True
                self._silence_start = None
            elif self._speech_started:
                # Speech ended, start silence timer
                if self._silence_start is None:
                    self._silence_start = time.time()
    
    def stop_recording(self) -> bytes:
        """Stop recording and return audio data."""
        with self._lock:
            self._is_recording = False
            audio_data = b''.join(self._recording)
            self._recording.clear()
        logger.debug(f"Recording stopped, {len(audio_data)} bytes")
        return audio_data
    
    def get_recording(self) -> bytes:
        """Get current recording without stopping."""
        with self._lock:
            return b''.join(self._recording)
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    @property
    def is_complete(self) -> bool:
        """
        Check if recording is complete.
        
        Complete when:
        - Speech was detected AND silence timeout reached, OR
        - Max recording duration reached
        """
        if not self._is_recording:
            return False
        
        now = time.time()
        
        # Check max duration
        if self._recording_start:
            duration = now - self._recording_start
            if duration >= self._config.max_recording_seconds:
                return True
        
        # Check silence timeout (only after speech started)
        if self._speech_started and self._silence_start:
            silence_duration = (now - self._silence_start) * 1000  # ms
            if silence_duration >= self._config.silence_duration_ms:
                return True
        
        return False
    
    @property
    def duration_seconds(self) -> float:
        """Get current recording duration."""
        if self._recording_start:
            return time.time() - self._recording_start
        return 0.0
    
    @property
    def has_speech(self) -> bool:
        """Check if speech was detected in recording."""
        return self._speech_started
