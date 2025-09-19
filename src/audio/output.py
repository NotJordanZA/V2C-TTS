"""
Audio output management for playing generated speech with buffering and queue management.
"""

import pyaudio
import numpy as np
import threading
import time
import logging
from typing import List, Optional, Callable
from collections import deque
import asyncio
from queue import Queue, Empty

from ..core.interfaces import (
    AudioOutputInterface, AudioDevice, PipelineConfig, 
    PipelineError, PipelineStage
)
from .device_manager import AudioDeviceManager

logger = logging.getLogger(__name__)


class AudioOutputBuffer:
    """Thread-safe buffer for audio output with smooth playback management."""
    
    def __init__(self, sample_rate: int = 16000, max_buffer_seconds: float = 5.0):
        self.sample_rate = sample_rate
        self.max_buffer_size = int(max_buffer_seconds * sample_rate)
        self._buffer = deque()
        self._lock = threading.Lock()
        self._total_samples = 0
    
    def write(self, audio_data: np.ndarray) -> None:
        """Add audio data to the output buffer."""
        with self._lock:
            # Convert to list and add to buffer
            audio_list = audio_data.flatten().tolist()
            
            # If buffer would exceed max size, remove old data
            while (len(self._buffer) + len(audio_list)) > self.max_buffer_size:
                if self._buffer:
                    self._buffer.popleft()
                else:
                    break
            
            self._buffer.extend(audio_list)
            self._total_samples += len(audio_list)
    
    def read(self, num_samples: int) -> np.ndarray:
        """Read audio data from buffer for playback."""
        with self._lock:
            if len(self._buffer) < num_samples:
                # Return available data + silence padding
                available_data = list(self._buffer)
                self._buffer.clear()
                
                # Pad with silence
                padding_needed = num_samples - len(available_data)
                result = available_data + [0.0] * padding_needed
            else:
                # Return requested amount
                result = []
                for _ in range(num_samples):
                    if self._buffer:
                        result.append(self._buffer.popleft())
                    else:
                        result.append(0.0)
            
            return np.array(result, dtype=np.float32)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
    
    @property
    def available_samples(self) -> int:
        """Get number of samples available in buffer."""
        with self._lock:
            return len(self._buffer)
    
    @property
    def available_duration(self) -> float:
        """Get duration of audio available in buffer (seconds)."""
        return self.available_samples / self.sample_rate
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0


class AudioOutputQueue:
    """Queue for managing multiple audio chunks to be played in sequence."""
    
    def __init__(self, max_queue_size: int = 10):
        self._queue = Queue(maxsize=max_queue_size)
        self._current_chunk: Optional[np.ndarray] = None
        self._current_position = 0
    
    def add_audio(self, audio_data: np.ndarray) -> bool:
        """
        Add audio data to the queue.
        
        Returns:
            bool: True if added successfully, False if queue is full
        """
        try:
            self._queue.put_nowait(audio_data.copy())
            return True
        except:
            logger.warning("Audio output queue is full, dropping audio chunk")
            return False
    
    def get_next_samples(self, num_samples: int) -> np.ndarray:
        """Get the next samples for playback, managing chunk transitions."""
        result = np.zeros(num_samples, dtype=np.float32)
        result_pos = 0
        
        while result_pos < num_samples:
            # If no current chunk, try to get next from queue
            if self._current_chunk is None or self._current_position >= len(self._current_chunk):
                try:
                    self._current_chunk = self._queue.get_nowait()
                    self._current_position = 0
                except Empty:
                    # No more audio available, fill rest with silence
                    break
            
            # Copy samples from current chunk
            if self._current_chunk is not None:
                samples_needed = num_samples - result_pos
                samples_available = len(self._current_chunk) - self._current_position
                samples_to_copy = min(samples_needed, samples_available)
                
                result[result_pos:result_pos + samples_to_copy] = \
                    self._current_chunk[self._current_position:self._current_position + samples_to_copy]
                
                result_pos += samples_to_copy
                self._current_position += samples_to_copy
        
        return result
    
    def clear(self) -> None:
        """Clear the queue and current chunk."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        
        self._current_chunk = None
        self._current_position = 0
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    @property
    def is_empty(self) -> bool:
        """Check if queue and current chunk are empty."""
        return (self._queue.empty() and 
                (self._current_chunk is None or 
                 self._current_position >= len(self._current_chunk)))


class AudioOutput(AudioOutputInterface):
    """Audio output management with buffering and queue management."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._pyaudio = None
        self._stream = None
        self._device_manager = None
        
        # Output management
        self._output_buffer = AudioOutputBuffer(
            sample_rate=config.sample_rate,
            max_buffer_seconds=5.0
        )
        self._output_queue = AudioOutputQueue(max_queue_size=20)
        
        # Playback control
        self._current_device_id: Optional[int] = None
        self._is_playing = False
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Audio format settings
        self._chunk_size = config.chunk_size
        self._sample_rate = config.sample_rate
    
    async def initialize(self) -> None:
        """Initialize the audio output system."""
        try:
            self._pyaudio = pyaudio.PyAudio()
            self._device_manager = AudioDeviceManager()
            self._device_manager.__enter__()
            self._initialized = True
            logger.info("AudioOutput initialized successfully")
        except Exception as e:
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                f"Failed to initialize audio output: {str(e)}",
                recoverable=False
            )
    
    async def cleanup(self) -> None:
        """Clean up audio output resources."""
        await self.stop_playback()
        
        if self._device_manager:
            self._device_manager.__exit__(None, None, None)
            self._device_manager = None
        
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        
        self._initialized = False
        logger.info("AudioOutput cleaned up")
    
    def get_available_devices(self) -> List[AudioDevice]:
        """Get list of available audio output devices."""
        if not self._device_manager:
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                "AudioOutput not initialized",
                recoverable=False
            )
        
        return self._device_manager.get_available_output_devices()
    
    async def play_audio(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Play audio through output device."""
        if not self._initialized:
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                "AudioOutput not initialized",
                recoverable=False
            )
        
        # Resample if necessary
        if sample_rate != self._sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, self._sample_rate)
        
        # Normalize audio to prevent clipping
        audio_data = self._normalize_audio(audio_data)
        
        # Add to queue for playback
        if not self._output_queue.add_audio(audio_data):
            logger.warning("Failed to add audio to output queue (queue full)")
        
        # Start playback if not already playing
        if not self._is_playing:
            await self._start_playback()
    
    async def stop_playback(self) -> None:
        """Stop current audio playback."""
        if not self._is_playing:
            return
        
        self._stop_event.set()
        self._is_playing = False
        
        # Stop and close stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error stopping audio stream: {e}")
            finally:
                self._stream = None
        
        # Wait for playback thread to finish
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)
            if self._playback_thread.is_alive():
                logger.warning("Playback thread did not stop gracefully")
        
        self._playback_thread = None
        self._current_device_id = None
        
        # Clear buffers
        self._output_buffer.clear()
        self._output_queue.clear()
        
        logger.info("Audio playback stopped")
    
    async def _start_playback(self, device_id: Optional[int] = None) -> None:
        """Start audio playback on specified device."""
        if device_id is None:
            # Use default output device
            default_device = self._device_manager.get_default_output_device()
            if default_device is None:
                raise PipelineError(
                    PipelineStage.AUDIO_OUTPUT,
                    "No default output device available",
                    recoverable=True
                )
            device_id = default_device.id
        
        # Validate device
        if not self._device_manager.validate_device(
            device_id, is_input=False, 
            sample_rate=self._sample_rate, 
            channels=1
        ):
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                f"Device {device_id} does not support required audio format",
                recoverable=True
            )
        
        self._current_device_id = device_id
        self._stop_event.clear()
        
        try:
            # Open audio stream
            self._stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self._sample_rate,
                output=True,
                output_device_index=device_id,
                frames_per_buffer=self._chunk_size,
                stream_callback=self._audio_output_callback
            )
            
            # Start playback thread
            self._playback_thread = threading.Thread(
                target=self._playback_worker,
                daemon=True
            )
            self._playback_thread.start()
            
            self._stream.start_stream()
            self._is_playing = True
            
            logger.info(f"Started audio playback on device {device_id}")
            
        except Exception as e:
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                f"Failed to start audio playback: {str(e)}",
                recoverable=True
            )
    
    def _audio_output_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback for audio output."""
        if status:
            logger.warning(f"Audio output stream status: {status}")
        
        # Get audio data from buffer
        audio_data = self._output_buffer.read(frame_count)
        
        # Convert to bytes
        output_data = audio_data.tobytes()
        
        return (output_data, pyaudio.paContinue)
    
    def _playback_worker(self) -> None:
        """Worker thread for managing audio playback queue."""
        while not self._stop_event.is_set():
            try:
                # Get next audio samples from queue
                chunk_samples = self._chunk_size * 2  # Buffer ahead
                audio_chunk = self._output_queue.get_next_samples(chunk_samples)
                
                if not np.all(audio_chunk == 0):  # If not all silence
                    # Add to output buffer
                    self._output_buffer.write(audio_chunk)
                
                # Sleep briefly to prevent tight loop
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in playback worker: {e}")
                time.sleep(0.1)  # Prevent tight error loop
        
        logger.debug("Playback worker thread stopped")
    
    def _resample_audio(self, audio_data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple resampling (basic implementation)."""
        if from_rate == to_rate:
            return audio_data
        
        # Simple linear interpolation resampling
        ratio = to_rate / from_rate
        new_length = int(len(audio_data) * ratio)
        
        # Create new time indices
        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio_data)
        
        return resampled.astype(np.float32)
    
    def _normalize_audio(self, audio_data: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Scale to target level
            scale_factor = target_level / max_val
            if scale_factor < 1.0:  # Only scale down, never amplify
                audio_data = audio_data * scale_factor
        
        return audio_data
    
    @property
    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        return self._is_playing
    
    @property
    def current_device_id(self) -> Optional[int]:
        """Get the currently selected output device ID."""
        return self._current_device_id
    
    def get_playback_status(self) -> dict:
        """Get current playback status for monitoring."""
        return {
            'is_playing': self._is_playing,
            'device_id': self._current_device_id,
            'buffer_samples': self._output_buffer.available_samples,
            'buffer_duration_seconds': self._output_buffer.available_duration,
            'queue_size': self._output_queue.queue_size,
            'queue_empty': self._output_queue.is_empty
        }
    
    def set_output_device(self, device_id: int) -> None:
        """Set the output device (will take effect on next playback start)."""
        if not self._device_manager.validate_device(
            device_id, is_input=False, 
            sample_rate=self._sample_rate, 
            channels=1
        ):
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                f"Device {device_id} does not support required audio format",
                recoverable=True
            )
        
        # If currently playing, we'll need to restart playback
        was_playing = self._is_playing
        if was_playing:
            asyncio.create_task(self.stop_playback())
        
        self._current_device_id = device_id
        
        if was_playing:
            asyncio.create_task(self._start_playback(device_id))
        
        logger.info(f"Output device set to {device_id}")