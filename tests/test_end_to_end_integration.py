"""
Comprehensive end-to-end integration tests for the voice character transformation pipeline.

This module tests the complete pipeline flow from audio input to character-transformed
speech output, including performance benchmarking and automated test scenarios.
"""

import pytest
import asyncio
import numpy as np
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.core.pipeline import VoicePipeline, PipelineState, PipelineMetrics
from src.core.config import AppConfig
from src.audio.capture import AudioCapture
from src.audio.output import AudioOutput
from src.stt.processor import STTProcessor
from src.character.transformer import CharacterTransformer
from src.tts.processor import TTSProcessor
from src.character.profile import CharacterProfile
from src.core.interfaces import AudioChunk


class SyntheticAudioGenerator:
    """Generates synthetic audio data for testing."""
    
    @staticmethod
    def generate_speech_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate synthetic speech-like audio data."""
        # Generate a simple sine wave with some noise to simulate speech
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        
        # Multiple frequency components to simulate speech
        frequencies = [200, 400, 800, 1600]  # Typical speech frequencies
        audio = np.zeros_like(t)
        
        for freq in frequencies:
            amplitude = np.random.uniform(0.1, 0.3)
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, len(t))
        audio += noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32)
    
    @staticmethod
    def generate_silence(duration_seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate silence for testing."""
        return np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)


class MockComponents:
    """Mock components for integration testing."""
    
    def __init__(self):
        self.stt_responses = ["Hello world", "How are you today", "This is a test"]
        self.character_responses = {
            "anime-waifu": ["Hello world desu~", "How are you today nya~", "This is a test kawaii~"],
            "patriotic-american": ["Hello there, fellow American!", "How are you doing, patriot?", "This is a great test!"],
            "slurring-drunk": ["Hellooo worrrldd", "Howww arrre youuu todayyy", "Thissss isss a tessst"]
        }
        self.current_response_index = 0
    
    def get_mock_stt_processor(self):
        """Create a mock STT processor."""
        mock_stt = Mock(spec=STTProcessor)
        mock_stt.process_audio = AsyncMock(side_effect=self._mock_stt_process)
        mock_stt.is_ready = Mock(return_value=True)
        return mock_stt
    
    def get_mock_character_transformer(self):
        """Create a mock character transformer."""
        mock_transformer = Mock(spec=CharacterTransformer)
        mock_transformer.transform_text = AsyncMock(side_effect=self._mock_transform_text)
        mock_transformer.load_character = Mock(side_effect=self._mock_load_character)
        return mock_transformer
    
    def get_mock_tts_processor(self):
        """Create a mock TTS processor."""
        mock_tts = Mock(spec=TTSProcessor)
        mock_tts.synthesize_speech = AsyncMock(side_effect=self._mock_synthesize_speech)
        mock_tts.is_ready = Mock(return_value=True)
        return mock_tts
    
    async def _mock_stt_process(self, audio_chunk: AudioChunk) -> str:
        """Mock STT processing with realistic delay."""
        await asyncio.sleep(0.1)  # Simulate processing time
        response = self.stt_responses[self.current_response_index % len(self.stt_responses)]
        return response
    
    async def _mock_transform_text(self, text: str, character: CharacterProfile) -> str:
        """Mock character transformation with realistic delay."""
        await asyncio.sleep(0.2)  # Simulate LLM processing time
        character_name = character.name
        if character_name in self.character_responses:
            responses = self.character_responses[character_name]
            response = responses[self.current_response_index % len(responses)]
            self.current_response_index += 1
            return response
        return f"{text} (transformed by {character_name})"
    
    async def _mock_synthesize_speech(self, text: str, voice_model) -> np.ndarray:
        """Mock TTS synthesis with realistic delay."""
        await asyncio.sleep(0.3)  # Simulate TTS processing time
        # Return synthetic audio data
        return SyntheticAudioGenerator.generate_speech_audio(duration_seconds=len(text) * 0.1)
    
    def _mock_load_character(self, character_name: str) -> CharacterProfile:
        """Mock character loading."""
        # Load actual character profile for realistic testing
        character_path = Path(f"characters/{character_name}.json")
        if character_path.exists():
            with open(character_path, 'r') as f:
                data = json.load(f)
            return CharacterProfile(**data)
        else:
            # Return a default character profile
            return CharacterProfile(
                name=character_name,
                description=f"Test character: {character_name}",
                personality_traits=["test"],
                speech_patterns={},
                vocabulary_preferences={},
                transformation_prompt=f"Transform text as {character_name}: {{text}}",
                voice_model_path=f"models/voices/{character_name}.pth"
            )


@pytest.mark.integration
class TestEndToEndIntegration:
    """Comprehensive end-to-end integration tests."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a test configuration."""
        from src.core.config import AppConfig, AudioConfig, STTConfig, CharacterConfig, TTSConfig, PerformanceConfig, LoggingConfig
        
        return AppConfig(
            audio=AudioConfig(
                sample_rate=16000,
                chunk_size=1024,
                input_device_id=0,
                output_device_id=0
            ),
            stt=STTConfig(
                model_size="base",
                device="cpu"
            ),
            character=CharacterConfig(
                profiles_dir="characters",
                default_character="default",
                llm_model_path="models/llm/test_model.bin"
            ),
            tts=TTSConfig(
                model_path="models/tts/test_model.pth",
                device="cpu",
                voice_models_dir="models/voices"
            ),
            performance=PerformanceConfig(
                max_latency_ms=2000,
                gpu_memory_fraction=0.8
            ),
            logging=LoggingConfig(
                level="WARNING"
            )
        )
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        return MockComponents()
    
    @pytest.fixture
    def synthetic_audio(self):
        """Generate synthetic audio for testing."""
        return SyntheticAudioGenerator()
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, mock_config, mock_components):
        """Test the complete pipeline from audio input to speech output."""
        # Create pipeline with mocked components
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(mock_config)
            
            # Initialize pipeline
            await pipeline.initialize()
            assert pipeline.get_state() == PipelineState.STOPPED
            
            # Start pipeline
            await pipeline.start()
            assert pipeline.get_state() == PipelineState.RUNNING
            
            # Process synthetic audio through the pipeline
            audio_data = SyntheticAudioGenerator.generate_speech_audio()
            audio_chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=16000,
                duration_ms=len(audio_data) / 16000 * 1000
            )
            
            # Process the audio chunk
            start_time = time.time()
            result = await pipeline.process_audio_chunk(audio_chunk)
            end_time = time.time()
            
            # Verify processing completed
            assert result is not None
            processing_time_ms = (end_time - start_time) * 1000
            assert processing_time_ms < mock_config.max_latency_ms
            
            # Stop pipeline
            await pipeline.stop()
            assert pipeline.get_state() == PipelineState.STOPPED
    
    @pytest.mark.asyncio
    async def test_all_character_profiles(self, mock_config, mock_components):
        """Test pipeline with all available character profiles."""
        character_profiles = ["anime-waifu", "patriotic-american", "slurring-drunk", "default"]
        
        for character_name in character_profiles:
            with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
                 patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
                 patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
                 patch('src.core.pipeline.AudioCapture'), \
                 patch('src.core.pipeline.AudioOutput'):
                
                pipeline = VoicePipeline(mock_config)
                await pipeline.initialize()
                
                # Set character
                await pipeline.set_character(character_name)
                
                # Start pipeline
                await pipeline.start()
                
                # Process audio
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=len(audio_data) / 16000 * 1000
                )
                
                result = await pipeline.process_audio_chunk(audio_chunk)
                assert result is not None
                
                # Verify character was applied
                current_character = pipeline.get_current_character()
                assert current_character.name == character_name
                
                await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, mock_config, mock_components):
        """Test performance benchmarking and latency requirements."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(mock_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # Test multiple audio chunks for performance measurement
            latencies = []
            num_tests = 10
            
            for i in range(num_tests):
                audio_data = SyntheticAudioGenerator.generate_speech_audio(duration_seconds=1.0)
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=1000
                )
                
                start_time = time.time()
                result = await pipeline.process_audio_chunk(audio_chunk)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                assert result is not None
                assert latency_ms < mock_config.max_latency_ms
            
            # Calculate performance metrics
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            # Verify performance requirements
            assert avg_latency < mock_config.max_latency_ms
            assert max_latency < mock_config.max_latency_ms * 1.5  # Allow 50% overhead for worst case
            
            # Get pipeline metrics
            metrics = pipeline.get_metrics()
            assert metrics.processed_chunks == num_tests
            assert metrics.successful_transformations == num_tests
            assert metrics.failed_transformations == 0
            
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_config, mock_components):
        """Test pipeline handling of concurrent audio processing."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(mock_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # Create multiple concurrent audio processing tasks
            async def process_audio_task(task_id: int):
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=len(audio_data) / 16000 * 1000
                )
                
                result = await pipeline.process_audio_chunk(audio_chunk)
                return task_id, result
            
            # Run concurrent tasks
            num_concurrent_tasks = 5
            tasks = [process_audio_task(i) for i in range(num_concurrent_tasks)]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Verify all tasks completed successfully
            assert len(results) == num_concurrent_tasks
            for task_id, result in results:
                assert result is not None
            
            # Verify concurrent processing didn't exceed reasonable time limits
            total_time_ms = (end_time - start_time) * 1000
            # Should be faster than sequential processing
            assert total_time_ms < mock_config.max_latency_ms * num_concurrent_tasks * 0.8
            
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, mock_config, mock_components):
        """Test error recovery mechanisms in the complete pipeline."""
        # Create components that will fail occasionally
        failing_stt = mock_components.get_mock_stt_processor()
        failing_stt.process_audio = AsyncMock(side_effect=[
            Exception("STT processing failed"),
            "Recovery successful",
            "Normal processing"
        ])
        
        with patch('src.core.pipeline.STTProcessor', return_value=failing_stt), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(mock_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # First audio chunk should fail and trigger error recovery
            audio_data = SyntheticAudioGenerator.generate_speech_audio()
            audio_chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=16000,
                duration_ms=len(audio_data) / 16000 * 1000
            )
            
            # Process should handle the error gracefully
            result1 = await pipeline.process_audio_chunk(audio_chunk)
            # First attempt might fail, but pipeline should remain stable
            
            # Second attempt should succeed due to error recovery
            result2 = await pipeline.process_audio_chunk(audio_chunk)
            assert result2 is not None
            
            # Third attempt should work normally
            result3 = await pipeline.process_audio_chunk(audio_chunk)
            assert result3 is not None
            
            # Verify pipeline is still running
            assert pipeline.get_state() == PipelineState.RUNNING
            
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, mock_config, mock_components):
        """Test memory usage monitoring during extended operation."""
        import psutil
        import os
        
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(mock_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # Monitor memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process many audio chunks to test for memory leaks
            num_chunks = 50
            for i in range(num_chunks):
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=len(audio_data) / 16000 * 1000
                )
                
                await pipeline.process_audio_chunk(audio_chunk)
                
                # Check memory every 10 chunks
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = current_memory - initial_memory
                    
                    # Memory increase should be reasonable (less than 100MB for testing)
                    assert memory_increase < 100, f"Memory usage increased by {memory_increase}MB"
            
            await pipeline.stop()


@pytest.mark.integration
class TestAutomatedTestSuite:
    """Automated test suite for continuous integration."""
    
    def test_character_profile_validation(self):
        """Validate all character profiles are properly formatted."""
        characters_dir = Path("characters")
        character_files = list(characters_dir.glob("*.json"))
        
        assert len(character_files) > 0, "No character profiles found"
        
        required_fields = [
            "name", "description", "personality_traits", "speech_patterns",
            "vocabulary_preferences", "transformation_prompt", "voice_model_path"
        ]
        
        for character_file in character_files:
            if character_file.name == ".gitkeep":
                continue
                
            with open(character_file, 'r') as f:
                character_data = json.load(f)
            
            # Validate required fields
            for field in required_fields:
                assert field in character_data, f"Missing field '{field}' in {character_file.name}"
            
            # Validate data types
            assert isinstance(character_data["name"], str)
            assert isinstance(character_data["description"], str)
            assert isinstance(character_data["personality_traits"], list)
            assert isinstance(character_data["speech_patterns"], dict)
            assert isinstance(character_data["vocabulary_preferences"], dict)
            assert isinstance(character_data["transformation_prompt"], str)
            assert isinstance(character_data["voice_model_path"], str)
    
    def test_configuration_validation(self):
        """Validate configuration files and settings."""
        config_files = [
            "config/default_config.yaml",
            "config/user_config.yaml",
            "config/user_config.json"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                # Basic validation that files can be loaded
                if config_file.endswith('.yaml'):
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                
                assert config_data is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization_speed(self):
        """Test that pipeline initialization meets performance requirements."""
        mock_config = AppConfig(
            audio_device_id=0,
            sample_rate=16000,
            chunk_size=1024,
            stt_model_size="base",
            llm_model_path="models/llm/test_model.bin",
            tts_model_path="models/tts/test_model.pth",
            gpu_device="cpu",
            max_latency_ms=2000
        )
        
        with patch('src.core.pipeline.STTProcessor'), \
             patch('src.core.pipeline.CharacterTransformer'), \
             patch('src.core.pipeline.TTSProcessor'), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            start_time = time.time()
            pipeline = VoicePipeline(mock_config)
            await pipeline.initialize()
            end_time = time.time()
            
            initialization_time_ms = (end_time - start_time) * 1000
            
            # Pipeline should initialize within 5 seconds
            assert initialization_time_ms < 5000, f"Pipeline initialization took {initialization_time_ms}ms"
    
    def test_model_file_structure(self):
        """Validate model file structure and organization."""
        models_dir = Path("models")
        
        # Check that models directory exists
        assert models_dir.exists(), "Models directory not found"
        
        # Check for voices subdirectory
        voices_dir = models_dir / "voices"
        assert voices_dir.exists(), "Voices directory not found"
        
        # Validate that .gitkeep files exist to maintain directory structure
        gitkeep_files = [
            models_dir / ".gitkeep",
            voices_dir / ".gitkeep" if voices_dir.exists() else None
        ]
        
        for gitkeep_file in gitkeep_files:
            if gitkeep_file:
                assert gitkeep_file.exists(), f"Missing .gitkeep file: {gitkeep_file}"


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])