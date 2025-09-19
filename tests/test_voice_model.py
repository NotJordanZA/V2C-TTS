"""
Unit tests for VoiceModelManager.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import soundfile as sf
import numpy as np

from src.tts.voice_model import VoiceModelManager
from src.core.interfaces import VoiceModel, PipelineError, PipelineStage


@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for voice models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def voice_model_manager(temp_models_dir):
    """Create a VoiceModelManager instance for testing."""
    return VoiceModelManager(str(temp_models_dir))


@pytest.fixture
def sample_voice_model():
    """Create a sample voice model for testing."""
    return VoiceModel(
        name="test_voice",
        model_path="/path/to/test_voice.wav",
        sample_rate=22050,
        language="en",
        gender="female"
    )


class TestVoiceModelManager:
    """Test cases for VoiceModelManager class."""
    
    def test_init(self, temp_models_dir):
        """Test VoiceModelManager initialization."""
        manager = VoiceModelManager(str(temp_models_dir))
        
        assert manager.models_directory == temp_models_dir
        assert manager.loaded_models == {}
        assert manager.model_cache == {}
        assert manager.supported_formats == {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        assert "default_female" in manager.default_models
        assert "default_male" in manager.default_models
    
    def test_initialize_creates_directory(self, temp_models_dir):
        """Test that initialize creates the models directory."""
        # Remove the directory
        temp_models_dir.rmdir()
        assert not temp_models_dir.exists()
        
        manager = VoiceModelManager(str(temp_models_dir))
        manager.initialize()
        
        assert temp_models_dir.exists()
        assert len(manager.loaded_models) >= 2  # At least default models
    
    def test_initialize_loads_default_models(self, voice_model_manager):
        """Test that initialize loads default models."""
        voice_model_manager.initialize()
        
        assert "default_female" in voice_model_manager.loaded_models
        assert "default_male" in voice_model_manager.loaded_models
        
        default_female = voice_model_manager.loaded_models["default_female"]
        assert default_female.language == "en"
        assert default_female.gender == "female"
    
    @patch('src.tts.voice_model.sf.SoundFile')
    def test_create_voice_model_from_file(self, mock_soundfile, voice_model_manager, temp_models_dir):
        """Test creating voice model from audio file."""
        # Create a test audio file with English name
        test_file = temp_models_dir / "english_test_voice.wav"
        test_file.touch()
        
        # Mock soundfile
        mock_audio = Mock()
        mock_audio.samplerate = 22050
        mock_audio.__len__ = Mock(return_value=66150)  # 3 seconds at 22050 Hz
        mock_soundfile.return_value.__enter__ = Mock(return_value=mock_audio)
        mock_soundfile.return_value.__exit__ = Mock(return_value=None)
        
        # Test
        voice_model = voice_model_manager._create_voice_model_from_file(test_file)
        
        assert voice_model.name == "english_test_voice"
        assert voice_model.model_path == str(test_file)
        assert voice_model.sample_rate == 22050
        assert voice_model.language == "en"
        assert voice_model.gender == "neutral"
    
    @patch('src.tts.voice_model.sf.SoundFile')
    def test_create_voice_model_from_file_short_audio(self, mock_soundfile, voice_model_manager, temp_models_dir):
        """Test creating voice model from short audio file."""
        test_file = temp_models_dir / "short_voice.wav"
        test_file.touch()
        
        # Mock soundfile with short duration
        mock_audio = Mock()
        mock_audio.samplerate = 22050
        mock_audio.__len__ = Mock(return_value=22050)  # 1 second
        mock_soundfile.return_value.__enter__ = Mock(return_value=mock_audio)
        mock_soundfile.return_value.__exit__ = Mock(return_value=None)
        
        # Should still create model but log warning
        voice_model = voice_model_manager._create_voice_model_from_file(test_file)
        assert voice_model.name == "short_voice"
    
    @patch('src.tts.voice_model.sf.SoundFile')
    def test_create_voice_model_from_invalid_file(self, mock_soundfile, voice_model_manager, temp_models_dir):
        """Test creating voice model from invalid audio file."""
        test_file = temp_models_dir / "invalid.wav"
        test_file.touch()
        
        # Mock soundfile to raise exception
        mock_soundfile.side_effect = Exception("Invalid audio file")
        
        with pytest.raises(ValueError, match="Invalid audio file"):
            voice_model_manager._create_voice_model_from_file(test_file)
    
    def test_load_voice_model_from_config_valid(self, voice_model_manager, temp_models_dir):
        """Test loading voice model from valid JSON config."""
        config_data = {
            "name": "test_config_voice",
            "model_path": "test_voice.wav",
            "sample_rate": 16000,
            "language": "es",
            "gender": "male"
        }
        
        config_file = temp_models_dir / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        voice_model = voice_model_manager._load_voice_model_from_config(config_file)
        
        assert voice_model is not None
        assert voice_model.name == "test_config_voice"
        assert voice_model.sample_rate == 16000
        assert voice_model.language == "es"
        assert voice_model.gender == "male"
    
    def test_load_voice_model_from_config_missing_required(self, voice_model_manager, temp_models_dir):
        """Test loading voice model from config missing required fields."""
        config_data = {
            "model_path": "test_voice.wav",
            "sample_rate": 16000
            # Missing 'name' and 'language'
        }
        
        config_file = temp_models_dir / "invalid_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        voice_model = voice_model_manager._load_voice_model_from_config(config_file)
        assert voice_model is None
    
    def test_load_voice_model_from_config_invalid_json(self, voice_model_manager, temp_models_dir):
        """Test loading voice model from invalid JSON config."""
        config_file = temp_models_dir / "invalid.json"
        with open(config_file, 'w') as f:
            f.write("invalid json content")
        
        voice_model = voice_model_manager._load_voice_model_from_config(config_file)
        assert voice_model is None
    
    def test_infer_language(self, voice_model_manager):
        """Test language inference from file path."""
        test_cases = [
            (Path("/path/to/spanish_voice.wav"), "es"),
            (Path("/path/to/french_voice.wav"), "fr"),  # Changed to have 'french' in filename
            (Path("/path/to/german_speaker.wav"), "de"),
            (Path("/path/to/english_voice.wav"), "en"),
            (Path("/path/to/unknown_voice.wav"), "en"),  # Default
        ]
        
        for file_path, expected_lang in test_cases:
            result = voice_model_manager._infer_language(file_path)
            assert result == expected_lang
    
    def test_infer_gender(self, voice_model_manager):
        """Test gender inference from file path."""
        test_cases = [
            (Path("/path/to/female_voice.wav"), "female"),
            (Path("/path/to/male_speaker.wav"), "male"),
            (Path("/path/to/woman_voice.wav"), "female"),
            (Path("/path/to/man_voice.wav"), "male"),
            (Path("/path/to/neutral_voice.wav"), "neutral"),
        ]
        
        for file_path, expected_gender in test_cases:
            result = voice_model_manager._infer_gender(file_path)
            assert result == expected_gender
    
    def test_get_voice_model(self, voice_model_manager, sample_voice_model):
        """Test getting voice model by name."""
        voice_model_manager.loaded_models["test_voice"] = sample_voice_model
        
        result = voice_model_manager.get_voice_model("test_voice")
        assert result == sample_voice_model
        
        result = voice_model_manager.get_voice_model("nonexistent")
        assert result is None
    
    def test_get_available_voices(self, voice_model_manager, sample_voice_model):
        """Test getting all available voices."""
        voice_model_manager.loaded_models["test_voice"] = sample_voice_model
        
        voices = voice_model_manager.get_available_voices()
        assert sample_voice_model in voices
    
    def test_get_voices_by_language(self, voice_model_manager):
        """Test filtering voices by language."""
        voice1 = VoiceModel("voice1", "", 22050, "en", "female")
        voice2 = VoiceModel("voice2", "", 22050, "es", "male")
        voice3 = VoiceModel("voice3", "", 22050, "en", "male")
        
        voice_model_manager.loaded_models.update({
            "voice1": voice1,
            "voice2": voice2,
            "voice3": voice3
        })
        
        en_voices = voice_model_manager.get_voices_by_language("en")
        assert len(en_voices) == 2
        assert voice1 in en_voices
        assert voice3 in en_voices
        
        es_voices = voice_model_manager.get_voices_by_language("es")
        assert len(es_voices) == 1
        assert voice2 in es_voices
    
    def test_get_voices_by_gender(self, voice_model_manager):
        """Test filtering voices by gender."""
        voice1 = VoiceModel("voice1", "", 22050, "en", "female")
        voice2 = VoiceModel("voice2", "", 22050, "es", "male")
        voice3 = VoiceModel("voice3", "", 22050, "en", "female")
        
        voice_model_manager.loaded_models.update({
            "voice1": voice1,
            "voice2": voice2,
            "voice3": voice3
        })
        
        female_voices = voice_model_manager.get_voices_by_gender("female")
        assert len(female_voices) == 2
        assert voice1 in female_voices
        assert voice3 in female_voices
        
        male_voices = voice_model_manager.get_voices_by_gender("male")
        assert len(male_voices) == 1
        assert voice2 in male_voices
    
    def test_add_voice_model(self, voice_model_manager, temp_models_dir):
        """Test adding a voice model."""
        # Create a real file for the test
        test_file = temp_models_dir / "test_voice.wav"
        test_file.touch()
        
        sample_voice_model = VoiceModel(
            name="test_voice",
            model_path=str(test_file),
            sample_rate=22050,
            language="en",
            gender="female"
        )
        
        voice_model_manager.add_voice_model(sample_voice_model)
        
        assert "test_voice" in voice_model_manager.loaded_models
        assert voice_model_manager.loaded_models["test_voice"] == sample_voice_model
    
    def test_add_voice_model_invalid_type(self, voice_model_manager):
        """Test adding invalid voice model type."""
        with pytest.raises(ValueError, match="voice_model must be a VoiceModel instance"):
            voice_model_manager.add_voice_model("not a voice model")
    
    def test_add_voice_model_missing_file(self, voice_model_manager):
        """Test adding voice model with missing file."""
        voice_model = VoiceModel("test", "/nonexistent/file.wav", 22050, "en", "female")
        
        with pytest.raises(FileNotFoundError):
            voice_model_manager.add_voice_model(voice_model)
    
    def test_remove_voice_model(self, voice_model_manager, sample_voice_model):
        """Test removing a voice model."""
        voice_model_manager.loaded_models["test_voice"] = sample_voice_model
        
        result = voice_model_manager.remove_voice_model("test_voice")
        assert result is True
        assert "test_voice" not in voice_model_manager.loaded_models
        
        result = voice_model_manager.remove_voice_model("nonexistent")
        assert result is False
    
    def test_validate_voice_model_valid(self, voice_model_manager):
        """Test validating a valid voice model."""
        voice_model = VoiceModel("test", "", 22050, "en", "female")
        assert voice_model_manager.validate_voice_model(voice_model) is True
    
    def test_validate_voice_model_invalid_name(self, voice_model_manager):
        """Test validating voice model with invalid name."""
        voice_model = VoiceModel("", "", 22050, "en", "female")
        assert voice_model_manager.validate_voice_model(voice_model) is False
    
    def test_validate_voice_model_invalid_sample_rate(self, voice_model_manager):
        """Test validating voice model with invalid sample rate."""
        voice_model = VoiceModel("test", "", -1, "en", "female")
        assert voice_model_manager.validate_voice_model(voice_model) is False
        
        voice_model = VoiceModel("test", "", 100000, "en", "female")
        assert voice_model_manager.validate_voice_model(voice_model) is False
    
    def test_validate_voice_model_invalid_gender(self, voice_model_manager):
        """Test validating voice model with invalid gender."""
        voice_model = VoiceModel("test", "", 22050, "en", "invalid_gender")
        assert voice_model_manager.validate_voice_model(voice_model) is False
    
    @patch('src.tts.voice_model.sf.SoundFile')
    def test_validate_voice_model_with_file(self, mock_soundfile, voice_model_manager, temp_models_dir):
        """Test validating voice model with audio file."""
        test_file = temp_models_dir / "test.wav"
        test_file.touch()
        
        # Mock valid audio file
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=1000)
        mock_soundfile.return_value.__enter__ = Mock(return_value=mock_audio)
        mock_soundfile.return_value.__exit__ = Mock(return_value=None)
        
        voice_model = VoiceModel("test", str(test_file), 22050, "en", "female")
        assert voice_model_manager.validate_voice_model(voice_model) is True
    
    def test_validate_voice_model_missing_file(self, voice_model_manager):
        """Test validating voice model with missing file."""
        voice_model = VoiceModel("test", "/nonexistent/file.wav", 22050, "en", "female")
        assert voice_model_manager.validate_voice_model(voice_model) is False
    
    def test_get_fallback_voice(self, voice_model_manager):
        """Test getting fallback voice model."""
        voice1 = VoiceModel("voice1", "", 22050, "en", "female")
        voice2 = VoiceModel("voice2", "", 22050, "es", "male")
        voice3 = VoiceModel("voice3", "", 22050, "en", "male")
        
        voice_model_manager.loaded_models.update({
            "voice1": voice1,
            "voice2": voice2,
            "voice3": voice3
        })
        
        # Test exact match
        fallback = voice_model_manager.get_fallback_voice("en", "female")
        assert fallback == voice1
        
        # Test language match only
        fallback = voice_model_manager.get_fallback_voice("es", "female")
        assert fallback == voice2
        
        # Test gender match only
        fallback = voice_model_manager.get_fallback_voice("fr", "male")
        assert fallback in [voice2, voice3]
        
        # Test any available
        fallback = voice_model_manager.get_fallback_voice("fr", "neutral")
        assert fallback in [voice1, voice2, voice3]
    
    def test_get_fallback_voice_empty(self, voice_model_manager):
        """Test getting fallback voice when no models loaded."""
        fallback = voice_model_manager.get_fallback_voice()
        assert fallback.name == "default_female"
    
    def test_save_voice_model_config(self, voice_model_manager, sample_voice_model, temp_models_dir):
        """Test saving voice model configuration."""
        config_path = voice_model_manager.save_voice_model_config(sample_voice_model)
        
        assert config_path.exists()
        assert config_path.name == "test_voice.json"
        
        # Verify content
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert config["name"] == sample_voice_model.name
        assert config["language"] == sample_voice_model.language
        assert config["gender"] == sample_voice_model.gender
    
    def test_save_voice_model_config_custom_path(self, voice_model_manager, sample_voice_model, temp_models_dir):
        """Test saving voice model configuration to custom path."""
        custom_path = temp_models_dir / "custom_config.json"
        
        result_path = voice_model_manager.save_voice_model_config(sample_voice_model, custom_path)
        
        assert result_path == custom_path
        assert custom_path.exists()
    
    def test_get_model_info(self, voice_model_manager):
        """Test getting model information."""
        voice1 = VoiceModel("voice1", "", 22050, "en", "female")
        voice2 = VoiceModel("voice2", "", 22050, "es", "male")
        
        voice_model_manager.loaded_models.update({
            "voice1": voice1,
            "voice2": voice2
        })
        
        info = voice_model_manager.get_model_info()
        
        assert info["total_models"] == 2
        assert "voice1" in info["models"]
        assert "voice2" in info["models"]
        assert "en" in info["by_language"]
        assert "es" in info["by_language"]
        assert "female" in info["by_gender"]
        assert "male" in info["by_gender"]
    
    @patch('src.tts.voice_model.VoiceModelManager._scan_voice_models')
    def test_initialize_scan_failure(self, mock_scan, voice_model_manager):
        """Test initialization when scanning fails."""
        mock_scan.side_effect = Exception("Scan failed")
        
        # Should raise PipelineError
        with pytest.raises(PipelineError) as exc_info:
            voice_model_manager.initialize()
        
        assert exc_info.value.stage == PipelineStage.TEXT_TO_SPEECH
        assert "Voice model manager initialization failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])