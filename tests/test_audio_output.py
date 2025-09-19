"""
Tests for audio output functionality including buffering, queue management, and playback.
"""

import pytest
import numpy as np
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import pyaudio

from src.audio.output import AudioOutput, AudioOutputBuffer, AudioOutputQueue
from src.core.interfaces import PipelineConfig, PipelineError, PipelineStage


class TestAudioOutputBuffer:
    """Test the audio output buffer implementation."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization with correct parameters."""
        buffer = AudioOutputBuffer(sample_rate=16000, max_buffer_seconds=3.0)
        
        assert buffer.sample_rate == 16000
        assert buffer.max_buffer_size == 48000  # 3 seconds * 16000 Hz
        assert buffer.available_samples == 0
        assert buffer.is_empty
    
    def test_write_and_read_data(self):
        """Test writing and reading audio data."""
        buffer = AudioOutputBuffer(sample_rate=1000, max_buffer_seconds=1.0)
        
        # Write some test data
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        buffer.write(test_data)
        
        assert buffer.available_samples == 5
        assert not buffer.is_empty
        
        # Read back the data
        read_data = buffer.read(5)
        np.testing.assert_array_equal(read_data, test_data)
        
        # Buffer should be empty now
        assert buffer.is_empty
    
    def test_read_with_padding(self):
        """Test reading more data than available (should pad with silence)."""
        buffer = AudioOutputBuffer(sample_rate=1000, max_buffer_seconds=1.0)
        
        # Write 3 samples
        test_data = np.array([1, 2, 3], dtype=np.float32)
        buffer.write(test_data)
        
        # Read 5 samples (should pad with 2 zeros)
        read_data = buffer.read(5)
        expected = np.array([1, 2, 3, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(read_data, expected)
        
        # Buffer should be empty
        assert buffer.is_empty
    
    def test_max_buffer_size_enforcement(self):
        """Test that buffer enforces maximum size."""
        buffer = AudioOutputBuffer(sample_rate=1000, max_buffer_seconds=0.005)  # 5 samples max
        
        # Write more data than buffer can hold
        data1 = np.array([1, 2, 3], dtype=np.float32)
        data2 = np.array([4, 5, 6, 7], dtype=np.float32)
        
        buffer.write(data1)
        buffer.write(data2)
        
        # Should only keep the most recent 5 samples
        assert buffer.available_samples == 5
        
        # Should contain [3, 4, 5, 6, 7] (oldest data dropped)
        read_data = buffer.read(5)
        expected = np.array([3, 4, 5, 6, 7], dtype=np.float32)
        np.testing.assert_array_equal(read_data, expected)
    
    def test_available_duration(self):
        """Test duration calculation."""
        buffer = AudioOutputBuffer(sample_rate=1000, max_buffer_seconds=1.0)
        
        # Write 500 samples (0.5 seconds at 1000 Hz)
        test_data = np.ones(500, dtype=np.float32)
        buffer.write(test_data)
        
        assert buffer.available_duration == 0.5
    
    def test_clear_buffer(self):
        """Test clearing the buffer."""
        buffer = AudioOutputBuffer(sample_rate=1000, max_buffer_seconds=1.0)
        
        # Write some data
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        buffer.write(test_data)
        
        assert not buffer.is_empty
        
        # Clear buffer
        buffer.clear()
        
        assert buffer.is_empty
        assert buffer.available_samples == 0


class TestAudioOutputQueue:
    """Test the audio output queue implementation."""
    
    def test_queue_initialization(self):
        """Test queue initialization."""
        queue = AudioOutputQueue(max_queue_size=5)
        
        assert queue.queue_size == 0
        assert queue.is_empty
    
    def test_add_and_get_audio(self):
        """Test adding and getting audio from queue."""
        queue = AudioOutputQueue(max_queue_size=5)
        
        # Add audio chunk
        audio_chunk = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        success = queue.add_audio(audio_chunk)
        
        assert success
        assert queue.queue_size == 1
        assert not queue.is_empty
        
        # Get samples
        samples = queue.get_next_samples(5)
        np.testing.assert_array_equal(samples, audio_chunk)
        
        # Queue should be empty now
        assert queue.is_empty
    
    def test_get_samples_across_chunks(self):
        """Test getting samples that span multiple chunks."""
        queue = AudioOutputQueue(max_queue_size=5)
        
        # Add two chunks
        chunk1 = np.array([1, 2, 3], dtype=np.float32)
        chunk2 = np.array([4, 5, 6, 7], dtype=np.float32)
        
        queue.add_audio(chunk1)
        queue.add_audio(chunk2)
        
        # Get 5 samples (should span both chunks)
        samples = queue.get_next_samples(5)
        expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        np.testing.assert_array_equal(samples, expected)
        
        # Get remaining samples
        remaining = queue.get_next_samples(5)
        expected_remaining = np.array([6, 7, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(remaining, expected_remaining)
    
    def test_queue_full_behavior(self):
        """Test behavior when queue is full."""
        queue = AudioOutputQueue(max_queue_size=2)
        
        # Fill queue
        chunk1 = np.array([1, 2], dtype=np.float32)
        chunk2 = np.array([3, 4], dtype=np.float32)
        chunk3 = np.array([5, 6], dtype=np.float32)  # This should fail
        
        assert queue.add_audio(chunk1)
        assert queue.add_audio(chunk2)
        assert not queue.add_audio(chunk3)  # Should fail (queue full)
        
        assert queue.queue_size == 2
    
    def test_clear_queue(self):
        """Test clearing the queue."""
        queue = AudioOutputQueue(max_queue_size=5)
        
        # Add some chunks
        queue.add_audio(np.array([1, 2, 3], dtype=np.float32))
        queue.add_audio(np.array([4, 5, 6], dtype=np.float32))
        
        assert not queue.is_empty
        
        # Clear queue
        queue.clear()
        
        assert queue.is_empty
        assert queue.queue_size == 0
    
    def test_partial_chunk_consumption(self):
        """Test consuming part of a chunk and continuing."""
        queue = AudioOutputQueue(max_queue_size=5)
        
        # Add a chunk
        chunk = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        queue.add_audio(chunk)
        
        # Get part of the chunk
        samples1 = queue.get_next_samples(3)
        expected1 = np.array([1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(samples1, expected1)
        
        # Get the rest
        samples2 = queue.get_next_samples(3)
        expected2 = np.array([4, 5, 6], dtype=np.float32)
        np.testing.assert_array_equal(samples2, expected2)
        
        # Should be empty now
        assert queue.is_empty


class TestAudioOutput:
    """Test the main AudioOutput class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return PipelineConfig(
            audio_device_id=0,
            sample_rate=16000,
            chunk_size=1024
        )
    
    @pytest.fixture
    def audio_output(self, config):
        """Create an AudioOutput instance."""
        return AudioOutput(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, audio_output):
        """Test AudioOutput initialization."""
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            with patch('src.audio.output.AudioDeviceManager') as mock_device_manager:
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                
                await audio_output.initialize()
                
                assert audio_output.is_initialized
                mock_pyaudio.assert_called_once()
                mock_device_manager.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, audio_output):
        """Test AudioOutput cleanup."""
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            with patch('src.audio.output.AudioDeviceManager') as mock_device_manager:
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                
                await audio_output.initialize()
                await audio_output.cleanup()
                
                assert not audio_output.is_initialized
                mock_manager_instance.__exit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_available_devices(self, audio_output):
        """Test getting available output devices."""
        with patch('pyaudio.PyAudio'):
            with patch('src.audio.output.AudioDeviceManager') as mock_device_manager:
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                mock_manager_instance.get_available_output_devices.return_value = []
                
                await audio_output.initialize()
                devices = audio_output.get_available_devices()
                
                mock_manager_instance.get_available_output_devices.assert_called_once()
                assert devices == []
    
    @pytest.mark.asyncio
    async def test_play_audio_not_initialized(self, audio_output):
        """Test that playing audio without initialization raises error."""
        audio_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        with pytest.raises(PipelineError) as exc_info:
            await audio_output.play_audio(audio_data, 16000)
        
        assert exc_info.value.stage == PipelineStage.AUDIO_OUTPUT
        assert "not initialized" in str(exc_info.value)
    
    def test_audio_normalization(self, audio_output):
        """Test audio normalization functionality."""
        # Test with audio that needs scaling down
        loud_audio = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
        normalized = audio_output._normalize_audio(loud_audio, target_level=0.8)
        
        # Should be scaled down
        assert np.max(np.abs(normalized)) <= 0.8
        
        # Test with audio that doesn't need scaling
        quiet_audio = np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32)
        normalized_quiet = audio_output._normalize_audio(quiet_audio, target_level=0.8)
        
        # Should remain unchanged (no amplification)
        np.testing.assert_array_equal(normalized_quiet, quiet_audio)
    
    def test_audio_resampling(self, audio_output):
        """Test audio resampling functionality."""
        # Test upsampling (8kHz to 16kHz)
        original_audio = np.array([1, 2, 3, 4], dtype=np.float32)
        resampled = audio_output._resample_audio(original_audio, 8000, 16000)
        
        # Should be twice as long
        assert len(resampled) == 8
        
        # Test downsampling (16kHz to 8kHz)
        resampled_down = audio_output._resample_audio(resampled, 16000, 8000)
        
        # Should be back to original length
        assert len(resampled_down) == 4
        
        # Test no resampling needed
        no_resample = audio_output._resample_audio(original_audio, 16000, 16000)
        np.testing.assert_array_equal(no_resample, original_audio)
    
    def test_get_playback_status(self, audio_output):
        """Test getting playback status information."""
        status = audio_output.get_playback_status()
        
        expected_keys = {
            'is_playing', 
            'device_id', 
            'buffer_samples', 
            'buffer_duration_seconds',
            'queue_size',
            'queue_empty'
        }
        assert set(status.keys()) == expected_keys
        assert status['is_playing'] == False
        assert status['device_id'] is None
    
    @pytest.mark.asyncio
    async def test_play_audio_with_resampling(self, audio_output):
        """Test playing audio that needs resampling."""
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            with patch('src.audio.output.AudioDeviceManager') as mock_device_manager:
                # Setup mocks
                mock_pyaudio_instance = Mock()
                mock_pyaudio.return_value = mock_pyaudio_instance
                
                mock_stream = Mock()
                mock_pyaudio_instance.open.return_value = mock_stream
                
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                mock_manager_instance.get_default_output_device.return_value = Mock(id=0)
                mock_manager_instance.validate_device.return_value = True
                
                await audio_output.initialize()
                
                # Play audio with different sample rate
                audio_data = np.array([1, 2, 3, 4], dtype=np.float32)
                await audio_output.play_audio(audio_data, 8000)  # Different from config (16000)
                
                # Should have started playback
                assert audio_output.is_playing
    
    @pytest.mark.asyncio
    async def test_stop_playback(self, audio_output):
        """Test stopping playback."""
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            with patch('src.audio.output.AudioDeviceManager') as mock_device_manager:
                # Setup mocks
                mock_pyaudio_instance = Mock()
                mock_pyaudio.return_value = mock_pyaudio_instance
                
                mock_stream = Mock()
                mock_pyaudio_instance.open.return_value = mock_stream
                
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                mock_manager_instance.get_default_output_device.return_value = Mock(id=0)
                mock_manager_instance.validate_device.return_value = True
                
                await audio_output.initialize()
                
                # Start playback
                audio_data = np.array([1, 2, 3, 4], dtype=np.float32)
                await audio_output.play_audio(audio_data, 16000)
                
                # Stop playback
                await audio_output.stop_playback()
                
                assert not audio_output.is_playing
                assert audio_output.current_device_id is None
                mock_stream.stop_stream.assert_called_once()
                mock_stream.close.assert_called_once()
    
    def test_audio_output_callback(self, audio_output):
        """Test the PyAudio output callback."""
        # Add some data to the buffer
        test_data = np.random.random(1024).astype(np.float32)
        audio_output._output_buffer.write(test_data)
        
        # Call the callback
        result, status = audio_output._audio_output_callback(None, 1024, None, None)
        
        # Should return continue signal
        assert status == pyaudio.paContinue
        assert result is not None
        assert len(result) == 1024 * 4  # 4 bytes per float32 sample
    
    def test_set_output_device_invalid(self, audio_output):
        """Test setting invalid output device."""
        with patch('pyaudio.PyAudio'):
            with patch('src.audio.output.AudioDeviceManager') as mock_device_manager:
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                mock_manager_instance.validate_device.return_value = False
                
                audio_output._device_manager = mock_manager_instance
                
                with pytest.raises(PipelineError) as exc_info:
                    audio_output.set_output_device(999)
                
                assert "does not support required audio format" in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__])