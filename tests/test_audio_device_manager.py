"""
Unit tests for audio device management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pyaudio

from src.audio.device_manager import AudioDeviceManager, get_audio_devices, get_default_devices
from src.core.interfaces import AudioDevice, PipelineError, PipelineStage


class TestAudioDeviceManager:
    """Test cases for AudioDeviceManager."""
    
    @pytest.fixture
    def mock_pyaudio(self):
        """Mock PyAudio instance."""
        mock = Mock(spec=pyaudio.PyAudio)
        return mock
    
    @pytest.fixture
    def sample_device_info(self):
        """Sample device information."""
        return {
            'index': 0,
            'name': 'Test Microphone',
            'maxInputChannels': 2,
            'maxOutputChannels': 0,
            'defaultSampleRate': 44100.0
        }
    
    @pytest.fixture
    def sample_output_device_info(self):
        """Sample output device information."""
        return {
            'index': 1,
            'name': 'Test Speakers',
            'maxInputChannels': 0,
            'maxOutputChannels': 2,
            'defaultSampleRate': 44100.0
        }
    
    def test_context_manager_initialization(self, mock_pyaudio):
        """Test that context manager properly initializes PyAudio."""
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                assert manager._pyaudio is not None
            
            mock_pyaudio.terminate.assert_called_once()
    
    def test_get_available_input_devices(self, mock_pyaudio, sample_device_info):
        """Test getting available input devices."""
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_device_info_by_index.return_value = sample_device_info
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                devices = manager.get_available_input_devices()
        
        assert len(devices) == 1
        device = devices[0]
        assert isinstance(device, AudioDevice)
        assert device.id == 0
        assert device.name == 'Test Microphone'
        assert device.channels == 2
        assert device.sample_rate == 44100
        assert device.is_input is True
    
    def test_get_available_output_devices(self, mock_pyaudio, sample_output_device_info):
        """Test getting available output devices."""
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_device_info_by_index.return_value = sample_output_device_info
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                devices = manager.get_available_output_devices()
        
        assert len(devices) == 1
        device = devices[0]
        assert isinstance(device, AudioDevice)
        assert device.id == 0  # Device ID comes from the loop index, not the device info
        assert device.name == 'Test Speakers'
        assert device.channels == 2
        assert device.sample_rate == 44100
        assert device.is_input is False
    
    def test_get_all_devices(self, mock_pyaudio, sample_device_info, sample_output_device_info):
        """Test getting all devices (input and output)."""
        # Mock the device enumeration for both input and output calls
        mock_pyaudio.get_device_count.return_value = 2
        
        # Create a side effect that cycles through the devices for multiple calls
        def mock_get_device_info(index):
            if index == 0:
                return sample_device_info
            elif index == 1:
                return sample_output_device_info
            else:
                raise Exception("Invalid device index")
        
        mock_pyaudio.get_device_info_by_index.side_effect = mock_get_device_info
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                devices = manager.get_all_devices()
        
        # Should have both input and output devices
        assert len(devices) == 2
        input_devices = [d for d in devices if d.is_input]
        output_devices = [d for d in devices if not d.is_input]
        assert len(input_devices) == 1
        assert len(output_devices) == 1
    
    def test_get_default_input_device(self, mock_pyaudio, sample_device_info):
        """Test getting default input device."""
        mock_pyaudio.get_default_input_device_info.return_value = sample_device_info
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                device = manager.get_default_input_device()
        
        assert device is not None
        assert device.name == 'Test Microphone'
        assert device.is_input is True
    
    def test_get_default_output_device(self, mock_pyaudio, sample_output_device_info):
        """Test getting default output device."""
        mock_pyaudio.get_default_output_device_info.return_value = sample_output_device_info
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                device = manager.get_default_output_device()
        
        assert device is not None
        assert device.name == 'Test Speakers'
        assert device.is_input is False
    
    def test_validate_device_success(self, mock_pyaudio, sample_device_info):
        """Test successful device validation."""
        mock_pyaudio.get_device_info_by_index.return_value = sample_device_info
        mock_pyaudio.is_format_supported.return_value = True
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                is_valid = manager.validate_device(0, is_input=True, sample_rate=16000, channels=1)
        
        assert is_valid is True
        mock_pyaudio.is_format_supported.assert_called_once()
    
    def test_validate_device_insufficient_channels(self, mock_pyaudio, sample_device_info):
        """Test device validation with insufficient channels."""
        # Device has 2 input channels, but we need 3
        sample_device_info['maxInputChannels'] = 2
        mock_pyaudio.get_device_info_by_index.return_value = sample_device_info
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                is_valid = manager.validate_device(0, is_input=True, sample_rate=16000, channels=3)
        
        assert is_valid is False
    
    def test_validate_device_format_not_supported(self, mock_pyaudio, sample_device_info):
        """Test device validation when format is not supported."""
        mock_pyaudio.get_device_info_by_index.return_value = sample_device_info
        mock_pyaudio.is_format_supported.return_value = False
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                is_valid = manager.validate_device(0, is_input=True, sample_rate=16000, channels=1)
        
        assert is_valid is False
    
    def test_get_device_info(self, mock_pyaudio, sample_device_info):
        """Test getting detailed device information."""
        mock_pyaudio.get_device_info_by_index.return_value = sample_device_info
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                info = manager.get_device_info(0)
        
        assert info == sample_device_info
    
    def test_error_when_not_initialized(self):
        """Test that methods raise errors when not initialized."""
        manager = AudioDeviceManager()
        
        with pytest.raises(PipelineError) as exc_info:
            manager.get_available_input_devices()
        
        assert exc_info.value.stage == PipelineStage.AUDIO_CAPTURE
        assert "not initialized" in str(exc_info.value)
    
    def test_enumeration_error_handling(self, mock_pyaudio):
        """Test error handling during device enumeration."""
        mock_pyaudio.get_device_count.side_effect = Exception("PyAudio error")
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                with pytest.raises(PipelineError) as exc_info:
                    manager.get_available_input_devices()
        
        assert exc_info.value.stage == PipelineStage.AUDIO_CAPTURE
        assert "Failed to enumerate input devices" in str(exc_info.value)
    
    def test_default_device_not_found(self, mock_pyaudio):
        """Test handling when default device is not found."""
        mock_pyaudio.get_default_input_device_info.side_effect = Exception("No default device")
        
        with patch('src.audio.device_manager.pyaudio.PyAudio', return_value=mock_pyaudio):
            with AudioDeviceManager() as manager:
                device = manager.get_default_input_device()
        
        assert device is None


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('src.audio.device_manager.AudioDeviceManager')
    def test_get_audio_devices(self, mock_manager_class):
        """Test get_audio_devices convenience function."""
        mock_manager = Mock()
        mock_manager_class.return_value.__enter__.return_value = mock_manager
        
        input_devices = [Mock(spec=AudioDevice)]
        output_devices = [Mock(spec=AudioDevice)]
        mock_manager.get_available_input_devices.return_value = input_devices
        mock_manager.get_available_output_devices.return_value = output_devices
        
        result_input, result_output = get_audio_devices()
        
        assert result_input == input_devices
        assert result_output == output_devices
    
    @patch('src.audio.device_manager.AudioDeviceManager')
    def test_get_default_devices(self, mock_manager_class):
        """Test get_default_devices convenience function."""
        mock_manager = Mock()
        mock_manager_class.return_value.__enter__.return_value = mock_manager
        
        default_input = Mock(spec=AudioDevice)
        default_output = Mock(spec=AudioDevice)
        mock_manager.get_default_input_device.return_value = default_input
        mock_manager.get_default_output_device.return_value = default_output
        
        result_input, result_output = get_default_devices()
        
        assert result_input == default_input
        assert result_output == default_output