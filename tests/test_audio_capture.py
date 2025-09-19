"""
Tests for audio capture functionality including real-time capture, 
circular buffering, and voice activity detection.
"""

import pytest
import numpy as np
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
import pyaudio

from src.audio.capture import AudioCapture, CircularAudioBuffer, VoiceActivityDetector
from src.core.interfaces import PipelineConfig, AudioChunk, PipelineError, PipelineStage


class TestCircularAudioBuffer:
    """Test the circular audio buffer implementation."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization with correct parameters."""
        buffer = CircularAudioBuffer(max_duration_seconds=5.0, sample_rate=16000)
        
        assert buffer.max_samples == 80000  # 5 seconds * 16000 Hz
        assert buffer.sample_rate == 16000
        assert buffer.current_size == 0
    
    def test_write_and_read_data(self):
        """Test writing and reading audio data."""
        buffer = CircularAudioBuffer(max_duration_seconds=1.0, sample_rate=1000)
        
        # Write some test data
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        buffer.write(test_data)
        
        assert buffer.current_size == 5
        
        # Read back the data
        read_data = buffer.read(5)
        np.testing.assert_array_equal(read_data, test_data)
    
    def test_circular_behavior(self):
        """Test that buffer behaves circularly when max size is exceeded."""
        buffer = CircularAudioBuffer(max_duration_seconds=0.005, sample_rate=1000)  # 5 samples max
        
        # Write more data than buffer can hold
        data1 = np.array([1, 2, 3], dtype=np.float32)
        data2 = np.array([4, 5, 6, 7], dtype=np.float32)
        
        buffer.write(data1)
        buffer.write(data2)
        
        # Should only keep the most recent 5 samples
        assert buffer.current_size == 5
        
        # Should contain [3, 4, 5, 6, 7] (oldest data dropped)
        read_data = buffer.read(5)
        expected = np.array([3, 4, 5, 6, 7], dtype=np.float32)
        np.testing.assert_array_equal(read_data, expected)
    
    def test_read_with_padding(self):
        """Test reading more data than available (should pad with zeros)."""
        buffer = CircularAudioBuffer(max_duration_seconds=1.0, sample_rate=1000)
        
        # Write 3 samples
        test_data = np.array([1, 2, 3], dtype=np.float32)
        buffer.write(test_data)
        
        # Read 5 samples (should pad with 2 zeros)
        read_data = buffer.read(5)
        expected = np.array([0, 0, 1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(read_data, expected)
    
    def test_get_recent_chunk(self):
        """Test getting a chunk of recent audio data by duration."""
        buffer = CircularAudioBuffer(max_duration_seconds=1.0, sample_rate=1000)
        
        # Write 1000 samples (1 second of data)
        test_data = np.arange(1000, dtype=np.float32)
        buffer.write(test_data)
        
        # Get 0.5 seconds (500 samples) of recent data
        chunk = buffer.get_recent_chunk(0.5)
        
        assert len(chunk) == 500
        # Should be the most recent 500 samples (500-999)
        expected = np.arange(500, 1000, dtype=np.float32)
        np.testing.assert_array_equal(chunk, expected)
    
    def test_clear_buffer(self):
        """Test clearing the buffer."""
        buffer = CircularAudioBuffer(max_duration_seconds=1.0, sample_rate=1000)
        
        # Write some data
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        buffer.write(test_data)
        
        assert buffer.current_size == 5
        
        # Clear buffer
        buffer.clear()
        
        assert buffer.current_size == 0
    
    def test_thread_safety(self):
        """Test that buffer operations are thread-safe."""
        buffer = CircularAudioBuffer(max_duration_seconds=1.0, sample_rate=1000)
        
        def writer_thread():
            for i in range(100):
                data = np.array([i], dtype=np.float32)
                buffer.write(data)
                time.sleep(0.001)
        
        def reader_thread():
            for _ in range(50):
                buffer.read(10)
                time.sleep(0.002)
        
        # Start threads
        writer = threading.Thread(target=writer_thread)
        reader = threading.Thread(target=reader_thread)
        
        writer.start()
        reader.start()
        
        writer.join()
        reader.join()
        
        # Should not crash and should have some data
        assert buffer.current_size > 0


class TestVoiceActivityDetector:
    """Test the voice activity detection implementation."""
    
    def test_vad_initialization(self):
        """Test VAD initialization with default parameters."""
        vad = VoiceActivityDetector()
        
        assert vad.energy_threshold == 0.01
        assert vad.zcr_threshold == 0.1
        assert vad.min_speech_duration == 0.1
        assert vad.min_silence_duration == 0.5
    
    def test_energy_calculation(self):
        """Test RMS energy calculation."""
        vad = VoiceActivityDetector()
        
        # Test with known signal
        signal = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        energy = vad._calculate_energy(signal)
        
        expected_energy = 1.0  # RMS of alternating +1/-1
        assert abs(energy - expected_energy) < 1e-6
    
    def test_zero_crossing_rate_calculation(self):
        """Test zero-crossing rate calculation."""
        vad = VoiceActivityDetector()
        
        # Signal that crosses zero 3 times: [1, -1, 1, -1]
        signal = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        zcr = vad._calculate_zcr(signal)
        
        # 6 zero crossings / (2 * (4-1) samples) = 6/6 = 1.0
        # np.diff(np.sign([1, -1, 1, -1])) = [-2, 2, -2]
        # np.abs([-2, 2, -2]) = [2, 2, 2]
        # np.sum([2, 2, 2]) = 6
        expected_zcr = 6.0 / (2 * 3)
        assert abs(zcr - expected_zcr) < 1e-6
    
    def test_speech_detection_silence(self):
        """Test that silence is correctly detected."""
        vad = VoiceActivityDetector(energy_threshold=0.1)
        
        # Low energy signal (silence)
        silence = np.zeros(1000, dtype=np.float32)
        
        is_speech, speech_ended = vad.detect_speech(silence, 0.0)
        
        assert not is_speech
        assert not speech_ended
    
    def test_speech_detection_speech(self):
        """Test that speech is correctly detected after minimum duration."""
        vad = VoiceActivityDetector(
            energy_threshold=0.01,
            zcr_threshold=0.1,
            min_speech_duration=0.05  # 50ms
        )
        
        # Generate speech-like signal (higher energy, moderate ZCR)
        t = np.linspace(0, 0.1, 1600)  # 100ms at 16kHz
        speech_signal = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
        
        # First detection - should not trigger immediately
        is_speech1, speech_ended1 = vad.detect_speech(speech_signal, 0.0)
        assert not is_speech1
        assert not speech_ended1
        
        # Second detection after minimum duration - should trigger
        is_speech2, speech_ended2 = vad.detect_speech(speech_signal, 0.06)
        assert is_speech2
        assert not speech_ended2
    
    def test_speech_end_detection(self):
        """Test that end of speech is correctly detected."""
        vad = VoiceActivityDetector(
            energy_threshold=0.01,
            zcr_threshold=0.1,
            min_speech_duration=0.05,
            min_silence_duration=0.1
        )
        
        # Start with speech
        t = np.linspace(0, 0.1, 1600)
        speech_signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        
        # Trigger speech detection
        vad.detect_speech(speech_signal, 0.0)
        vad.detect_speech(speech_signal, 0.06)  # Should start speech
        
        # Now silence
        silence = np.zeros(1600, dtype=np.float32)
        
        # First silence detection - should not end speech immediately
        is_speech1, speech_ended1 = vad.detect_speech(silence, 0.12)
        assert is_speech1  # Still in speech
        assert not speech_ended1
        
        # After minimum silence duration - should end speech
        is_speech2, speech_ended2 = vad.detect_speech(silence, 0.25)
        assert not is_speech2
        assert speech_ended2
    
    def test_sensitivity_adjustment(self):
        """Test sensitivity adjustment affects thresholds."""
        vad = VoiceActivityDetector()
        original_threshold = vad.energy_threshold
        
        # Higher sensitivity should lower threshold
        vad.set_sensitivity(0.8)
        high_sens_threshold = vad.energy_threshold
        assert high_sens_threshold < original_threshold
        
        # Lower sensitivity should raise threshold
        vad.set_sensitivity(0.2)
        low_sens_threshold = vad.energy_threshold
        assert low_sens_threshold > original_threshold
        assert low_sens_threshold > high_sens_threshold


class TestAudioCapture:
    """Test the main AudioCapture class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return PipelineConfig(
            audio_device_id=0,
            sample_rate=16000,
            chunk_size=1024
        )
    
    @pytest.fixture
    def audio_capture(self, config):
        """Create an AudioCapture instance."""
        return AudioCapture(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, audio_capture):
        """Test AudioCapture initialization."""
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            with patch('src.audio.capture.AudioDeviceManager') as mock_device_manager:
                await audio_capture.initialize()
                
                assert audio_capture.is_initialized
                mock_pyaudio.assert_called_once()
                mock_device_manager.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, audio_capture):
        """Test AudioCapture cleanup."""
        # Initialize first
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            with patch('src.audio.capture.AudioDeviceManager') as mock_device_manager:
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                
                await audio_capture.initialize()
                await audio_capture.cleanup()
                
                assert not audio_capture.is_initialized
                mock_manager_instance.__exit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_available_devices(self, audio_capture):
        """Test getting available devices."""
        with patch('pyaudio.PyAudio'):
            with patch('src.audio.capture.AudioDeviceManager') as mock_device_manager:
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                mock_manager_instance.get_available_input_devices.return_value = []
                
                await audio_capture.initialize()
                devices = audio_capture.get_available_devices()
                
                mock_manager_instance.get_available_input_devices.assert_called_once()
                assert devices == []
    
    @pytest.mark.asyncio
    async def test_start_capture_not_initialized(self, audio_capture):
        """Test that starting capture without initialization raises error."""
        callback = Mock()
        
        with pytest.raises(PipelineError) as exc_info:
            await audio_capture.start_capture(0, callback)
        
        assert exc_info.value.stage == PipelineStage.AUDIO_CAPTURE
        assert "not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_start_capture_invalid_device(self, audio_capture):
        """Test starting capture with invalid device."""
        with patch('pyaudio.PyAudio'):
            with patch('src.audio.capture.AudioDeviceManager') as mock_device_manager:
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                mock_manager_instance.validate_device.return_value = False
                
                await audio_capture.initialize()
                
                callback = Mock()
                with pytest.raises(PipelineError) as exc_info:
                    await audio_capture.start_capture(999, callback)
                
                assert "does not support required audio format" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_start_capture_success(self, audio_capture):
        """Test successful capture start."""
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            with patch('src.audio.capture.AudioDeviceManager') as mock_device_manager:
                # Setup mocks
                mock_pyaudio_instance = Mock()
                mock_pyaudio.return_value = mock_pyaudio_instance
                
                mock_stream = Mock()
                mock_pyaudio_instance.open.return_value = mock_stream
                
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                mock_manager_instance.validate_device.return_value = True
                
                await audio_capture.initialize()
                
                callback = Mock()
                await audio_capture.start_capture(0, callback)
                
                assert audio_capture.is_capturing
                assert audio_capture.current_device_id == 0
                mock_stream.start_stream.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_capture(self, audio_capture):
        """Test stopping capture."""
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            with patch('src.audio.capture.AudioDeviceManager') as mock_device_manager:
                # Setup mocks
                mock_pyaudio_instance = Mock()
                mock_pyaudio.return_value = mock_pyaudio_instance
                
                mock_stream = Mock()
                mock_pyaudio_instance.open.return_value = mock_stream
                
                mock_manager_instance = Mock()
                mock_manager_instance.__enter__ = Mock(return_value=mock_manager_instance)
                mock_manager_instance.__exit__ = Mock(return_value=None)
                mock_device_manager.return_value = mock_manager_instance
                mock_manager_instance.validate_device.return_value = True
                
                await audio_capture.initialize()
                
                # Start capture
                callback = Mock()
                await audio_capture.start_capture(0, callback)
                
                # Stop capture
                await audio_capture.stop_capture()
                
                assert not audio_capture.is_capturing
                assert audio_capture.current_device_id is None
                mock_stream.stop_stream.assert_called_once()
                mock_stream.close.assert_called_once()
    
    def test_set_sensitivity(self, audio_capture):
        """Test setting VAD sensitivity."""
        # Valid sensitivity
        audio_capture.set_sensitivity(0.5)
        
        # Invalid sensitivity should raise ValueError
        with pytest.raises(ValueError):
            audio_capture.set_sensitivity(1.5)
        
        with pytest.raises(ValueError):
            audio_capture.set_sensitivity(-0.1)
    
    def test_audio_stream_callback(self, audio_capture):
        """Test the PyAudio stream callback."""
        # Create mock audio data
        audio_data = np.random.random(1024).astype(np.float32)
        in_data = audio_data.tobytes()
        
        # Call the callback
        result = audio_capture._audio_stream_callback(in_data, 1024, None, None)
        
        # Should return continue signal
        assert result == (None, pyaudio.paContinue)
        
        # Should have added data to buffer
        assert audio_capture._circular_buffer.current_size == 1024
    
    def test_get_buffer_status(self, audio_capture):
        """Test getting buffer status information."""
        status = audio_capture.get_buffer_status()
        
        expected_keys = {
            'buffer_size_samples', 
            'buffer_duration_seconds', 
            'is_capturing', 
            'device_id'
        }
        assert set(status.keys()) == expected_keys
        assert status['is_capturing'] == False
        assert status['device_id'] is None
    
    @pytest.mark.asyncio
    async def test_capture_worker_voice_activity(self, audio_capture):
        """Test that capture worker processes voice activity correctly."""
        # This is a more complex integration test
        callback_calls = []
        
        def test_callback(chunk: AudioChunk):
            callback_calls.append(chunk)
        
        # Mock the VAD to simulate speech ending
        with patch.object(audio_capture._vad, 'detect_speech') as mock_vad:
            mock_vad.side_effect = [
                (False, False),  # No speech
                (True, False),   # Speech started
                (True, True),    # Speech ended
            ]
            
            # Add some test data to buffer
            test_data = np.random.random(4800).astype(np.float32)  # 0.3 seconds at 16kHz
            audio_capture._circular_buffer.write(test_data)
            
            # Set callback
            audio_capture._audio_callback = test_callback
            
            # Run worker for a short time
            audio_capture._stop_event.clear()
            worker_thread = threading.Thread(target=audio_capture._capture_worker)
            worker_thread.start()
            
            # Let it run briefly
            time.sleep(0.35)
            
            # Stop worker
            audio_capture._stop_event.set()
            worker_thread.join(timeout=1.0)
            
            # Should have called callback when speech ended
            assert len(callback_calls) > 0
            assert isinstance(callback_calls[0], AudioChunk)


if __name__ == '__main__':
    pytest.main([__file__])