"""
Audio device management for enumerating and managing audio input/output devices.
"""

import pyaudio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from ..core.interfaces import AudioDevice, PipelineError, PipelineStage

logger = logging.getLogger(__name__)


class AudioDeviceManager:
    """Manages audio device enumeration and validation."""
    
    def __init__(self):
        self._pyaudio = None
        self._device_cache: Optional[List[AudioDevice]] = None
        
    def __enter__(self):
        """Context manager entry."""
        self._pyaudio = pyaudio.PyAudio()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
    
    def get_available_input_devices(self) -> List[AudioDevice]:
        """Get list of available audio input devices."""
        if not self._pyaudio:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                "AudioDeviceManager not initialized. Use as context manager.",
                recoverable=False
            )
        
        devices = []
        try:
            device_count = self._pyaudio.get_device_count()
            
            for i in range(device_count):
                device_info = self._pyaudio.get_device_info_by_index(i)
                
                # Only include input devices
                if device_info['maxInputChannels'] > 0:
                    audio_device = AudioDevice(
                        id=i,
                        name=device_info['name'],
                        channels=device_info['maxInputChannels'],
                        sample_rate=int(device_info['defaultSampleRate']),
                        is_input=True
                    )
                    devices.append(audio_device)
                    
        except Exception as e:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                f"Failed to enumerate input devices: {str(e)}",
                recoverable=True
            )
        
        logger.info(f"Found {len(devices)} input devices")
        return devices
    
    def get_available_output_devices(self) -> List[AudioDevice]:
        """Get list of available audio output devices."""
        if not self._pyaudio:
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                "AudioDeviceManager not initialized. Use as context manager.",
                recoverable=False
            )
        
        devices = []
        try:
            device_count = self._pyaudio.get_device_count()
            
            for i in range(device_count):
                device_info = self._pyaudio.get_device_info_by_index(i)
                
                # Only include output devices
                if device_info['maxOutputChannels'] > 0:
                    audio_device = AudioDevice(
                        id=i,
                        name=device_info['name'],
                        channels=device_info['maxOutputChannels'],
                        sample_rate=int(device_info['defaultSampleRate']),
                        is_input=False
                    )
                    devices.append(audio_device)
                    
        except Exception as e:
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                f"Failed to enumerate output devices: {str(e)}",
                recoverable=True
            )
        
        logger.info(f"Found {len(devices)} output devices")
        return devices
    
    def get_all_devices(self) -> List[AudioDevice]:
        """Get all available audio devices (input and output)."""
        input_devices = self.get_available_input_devices()
        output_devices = self.get_available_output_devices()
        return input_devices + output_devices
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        """Get the default input device."""
        if not self._pyaudio:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                "AudioDeviceManager not initialized. Use as context manager.",
                recoverable=False
            )
        
        try:
            default_info = self._pyaudio.get_default_input_device_info()
            return AudioDevice(
                id=default_info['index'],
                name=default_info['name'],
                channels=default_info['maxInputChannels'],
                sample_rate=int(default_info['defaultSampleRate']),
                is_input=True
            )
        except Exception as e:
            logger.warning(f"No default input device found: {str(e)}")
            return None
    
    def get_default_output_device(self) -> Optional[AudioDevice]:
        """Get the default output device."""
        if not self._pyaudio:
            raise PipelineError(
                PipelineStage.AUDIO_OUTPUT,
                "AudioDeviceManager not initialized. Use as context manager.",
                recoverable=False
            )
        
        try:
            default_info = self._pyaudio.get_default_output_device_info()
            return AudioDevice(
                id=default_info['index'],
                name=default_info['name'],
                channels=default_info['maxOutputChannels'],
                sample_rate=int(default_info['defaultSampleRate']),
                is_input=False
            )
        except Exception as e:
            logger.warning(f"No default output device found: {str(e)}")
            return None
    
    def validate_device(self, device_id: int, is_input: bool, 
                       sample_rate: int = 16000, channels: int = 1) -> bool:
        """Validate that a device supports the specified configuration."""
        if not self._pyaudio:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE if is_input else PipelineStage.AUDIO_OUTPUT,
                "AudioDeviceManager not initialized. Use as context manager.",
                recoverable=False
            )
        
        try:
            # Check if device exists
            device_info = self._pyaudio.get_device_info_by_index(device_id)
            
            # Check if device supports the required direction
            if is_input and device_info['maxInputChannels'] < channels:
                return False
            if not is_input and device_info['maxOutputChannels'] < channels:
                return False
            
            # Test if the format is supported
            format_info = {
                'rate': sample_rate,
                'channels': channels,
                'format': pyaudio.paInt16,
                'input': is_input,
                'output': not is_input,
                'input_device_index': device_id if is_input else None,
                'output_device_index': device_id if not is_input else None
            }
            
            return self._pyaudio.is_format_supported(**format_info)
            
        except Exception as e:
            logger.warning(f"Device validation failed for device {device_id}: {str(e)}")
            return False
    
    def get_device_info(self, device_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific device."""
        if not self._pyaudio:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                "AudioDeviceManager not initialized. Use as context manager.",
                recoverable=False
            )
        
        try:
            return self._pyaudio.get_device_info_by_index(device_id)
        except Exception as e:
            raise PipelineError(
                PipelineStage.AUDIO_CAPTURE,
                f"Failed to get device info for device {device_id}: {str(e)}",
                recoverable=True
            )


def get_audio_devices() -> tuple[List[AudioDevice], List[AudioDevice]]:
    """Convenience function to get input and output devices."""
    with AudioDeviceManager() as manager:
        input_devices = manager.get_available_input_devices()
        output_devices = manager.get_available_output_devices()
        return input_devices, output_devices


def get_default_devices() -> tuple[Optional[AudioDevice], Optional[AudioDevice]]:
    """Convenience function to get default input and output devices."""
    with AudioDeviceManager() as manager:
        default_input = manager.get_default_input_device()
        default_output = manager.get_default_output_device()
        return default_input, default_output