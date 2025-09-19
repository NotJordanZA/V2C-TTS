"""
Tests for voice model integration and character voice mapping.

This module tests voice model loading, validation, and the integration
between character profiles and voice models.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.tts.voice_model import VoiceModelManager
from src.tts.character_voice_mapper import CharacterVoiceMapper, VoiceMapping
from src.character.profile import CharacterProfileManager
from src.core.interfaces import VoiceModel, PipelineError


class TestVoiceModelManager:
    """Test voice model manager functionality."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory for testing."""
        temp_dir = tempfile.mkdtemp()
        models_dir = Path(temp_dir) / "models" / "voices"
        models_dir.mkdir(parents=True)
        
        yield str(models_dir)
        
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def voice_manager(self, temp_models_dir):
        """Create voice model manager with temporary directory."""
        return VoiceModelManager(temp_models_dir)
    
    def test_voice_manager_initialization(self, voice_manager):
        """Test that voice manager initializes correctly."""
        voice_manager.initialize()
        
        # Should have default models
        available_voices = voice_manager.get_available_voices()
        voice_names = [voice.name for voice in available_voices]
        
        assert "default_female" in voice_names
        assert "default_male" in voice_names
    
    def test_voice_model_validation(self, voice_manager):
        """Test voice model validation."""
        voice_manager.initialize()
        
        # Valid voice model
        valid_voice = VoiceModel(
            name="test_voice",
            model_path="",
            sample_rate=22050,
            language="en",
            gender="female"
        )
        
        assert voice_manager.validate_voice_model(valid_voice)
        
        # Invalid voice model - bad sample rate
        invalid_voice = VoiceModel(
            name="bad_voice",
            model_path="",
            sample_rate=-1,
            language="en",
            gender="female"
        )
        
        assert not voice_manager.validate_voice_model(invalid_voice)
        
        # Invalid voice model - bad gender
        invalid_voice2 = VoiceModel(
            name="bad_voice2",
            model_path="",
            sample_rate=22050,
            language="en",
            gender="invalid_gender"
        )
        
        assert not voice_manager.validate_voice_model(invalid_voice2)
    
    def test_add_and_remove_voice_model(self, voice_manager):
        """Test adding and removing voice models."""
        voice_manager.initialize()
        
        # Add voice model
        test_voice = VoiceModel(
            name="test_voice",
            model_path="",
            sample_rate=22050,
            language="en",
            gender="female"
        )
        
        voice_manager.add_voice_model(test_voice)
        
        # Check it was added
        retrieved_voice = voice_manager.get_voice_model("test_voice")
        assert retrieved_voice is not None
        assert retrieved_voice.name == "test_voice"
        
        # Remove voice model
        removed = voice_manager.remove_voice_model("test_voice")
        assert removed
        
        # Check it was removed
        retrieved_voice = voice_manager.get_voice_model("test_voice")
        assert retrieved_voice is None
    
    def test_get_voices_by_language(self, voice_manager):
        """Test filtering voices by language."""
        voice_manager.initialize()
        
        # Add test voices with different languages
        voice_manager.add_voice_model(VoiceModel("en_voice", "", 22050, "en", "female"))
        voice_manager.add_voice_model(VoiceModel("es_voice", "", 22050, "es", "male"))
        
        en_voices = voice_manager.get_voices_by_language("en")
        es_voices = voice_manager.get_voices_by_language("es")
        
        en_names = [voice.name for voice in en_voices]
        es_names = [voice.name for voice in es_voices]
        
        assert "en_voice" in en_names
        assert "default_female" in en_names  # Default voices are English
        assert "default_male" in en_names
        
        assert "es_voice" in es_names
        assert "en_voice" not in es_names
    
    def test_get_voices_by_gender(self, voice_manager):
        """Test filtering voices by gender."""
        voice_manager.initialize()
        
        female_voices = voice_manager.get_voices_by_gender("female")
        male_voices = voice_manager.get_voices_by_gender("male")
        
        female_names = [voice.name for voice in female_voices]
        male_names = [voice.name for voice in male_voices]
        
        assert "default_female" in female_names
        assert "default_male" in male_names
        assert "default_female" not in male_names
        assert "default_male" not in female_names
    
    def test_fallback_voice_selection(self, voice_manager):
        """Test fallback voice selection logic."""
        voice_manager.initialize()
        
        # Add test voices
        voice_manager.add_voice_model(VoiceModel("en_female", "", 22050, "en", "female"))
        voice_manager.add_voice_model(VoiceModel("en_male", "", 22050, "en", "male"))
        voice_manager.add_voice_model(VoiceModel("es_female", "", 22050, "es", "female"))
        
        # Test exact match
        fallback = voice_manager.get_fallback_voice("en", "female")
        assert fallback.language == "en"
        assert fallback.gender == "female"
        
        # Test language match only
        fallback = voice_manager.get_fallback_voice("en", "neutral")
        assert fallback.language == "en"
        
        # Test gender match only
        fallback = voice_manager.get_fallback_voice("fr", "female")
        assert fallback.gender == "female"
        
        # Test no match - should return any available voice
        fallback = voice_manager.get_fallback_voice("fr", "neutral")
        assert fallback is not None
    
    def test_voice_model_config_loading(self, voice_manager, temp_models_dir):
        """Test loading voice models from JSON configuration files."""
        # Create test config file
        config_data = {
            "name": "test_config_voice",
            "model_path": "",
            "sample_rate": 22050,
            "language": "en",
            "gender": "female"
        }
        
        config_path = Path(temp_models_dir) / "test_config_voice.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        voice_manager.initialize()
        
        # Check that voice was loaded from config
        voice = voice_manager.get_voice_model("test_config_voice")
        assert voice is not None
        assert voice.name == "test_config_voice"
        assert voice.language == "en"
        assert voice.gender == "female"
    
    def test_model_info_generation(self, voice_manager):
        """Test model information generation."""
        voice_manager.initialize()
        
        # Add test voices
        voice_manager.add_voice_model(VoiceModel("en_female", "", 22050, "en", "female"))
        voice_manager.add_voice_model(VoiceModel("es_male", "", 22050, "es", "male"))
        
        info = voice_manager.get_model_info()
        
        assert "total_models" in info
        assert "by_language" in info
        assert "by_gender" in info
        assert "models" in info
        
        assert info["total_models"] >= 4  # At least default + test voices
        assert "en" in info["by_language"]
        assert "es" in info["by_language"]
        assert "female" in info["by_gender"]
        assert "male" in info["by_gender"]


class TestCharacterVoiceMapper:
    """Test character voice mapping functionality."""
    
    @pytest.fixture
    def character_manager(self):
        """Create mock character profile manager."""
        manager = Mock(spec=CharacterProfileManager)
        return manager
    
    @pytest.fixture
    def voice_manager(self):
        """Create mock voice model manager."""
        manager = Mock(spec=VoiceModelManager)
        
        # Mock available voices
        voices = [
            VoiceModel("default_female", "", 22050, "en", "female"),
            VoiceModel("default_male", "", 22050, "en", "male"),
            VoiceModel("anime_waifu", "path/to/anime.wav", 22050, "en", "female"),
            VoiceModel("patriotic_american", "path/to/patriotic.wav", 22050, "en", "male"),
            VoiceModel("slurring_drunk", "path/to/drunk.wav", 22050, "en", "neutral")
        ]
        
        manager.get_available_voices.return_value = voices
        manager.get_voice_model.side_effect = lambda name: next(
            (v for v in voices if v.name == name), None
        )
        manager.get_fallback_voice.return_value = voices[0]  # default_female
        
        return manager
    
    @pytest.fixture
    def voice_mapper(self, character_manager, voice_manager):
        """Create character voice mapper."""
        return CharacterVoiceMapper(character_manager, voice_manager)
    
    def test_mapper_initialization(self, voice_mapper):
        """Test that character voice mapper initializes correctly."""
        voice_mapper.initialize()
        
        # Should have default mappings
        assert "default" in voice_mapper.mappings
        assert "anime-waifu" in voice_mapper.mappings
        assert "patriotic-american" in voice_mapper.mappings
        assert "slurring-drunk" in voice_mapper.mappings
    
    def test_get_voice_for_character(self, voice_mapper):
        """Test getting voice model for character."""
        voice_mapper.initialize()
        
        # Test anime waifu character
        voice = voice_mapper.get_voice_for_character("anime-waifu")
        assert voice is not None
        assert voice.name == "anime_waifu"
        
        # Test patriotic american character
        voice = voice_mapper.get_voice_for_character("patriotic-american")
        assert voice is not None
        assert voice.name == "patriotic_american"
        
        # Test slurring drunk character
        voice = voice_mapper.get_voice_for_character("slurring-drunk")
        assert voice is not None
        assert voice.name == "slurring_drunk"
        
        # Test default character
        voice = voice_mapper.get_voice_for_character("default")
        assert voice is not None
        assert voice.name in ["default_female", "default_male"]
    
    def test_get_voice_for_unknown_character(self, voice_mapper):
        """Test getting voice for unknown character returns fallback."""
        voice_mapper.initialize()
        
        voice = voice_mapper.get_voice_for_character("unknown_character")
        assert voice is not None
        assert voice.name == "default_female"  # Fallback voice
    
    def test_custom_mapping(self, voice_mapper):
        """Test custom character to voice mapping."""
        voice_mapper.initialize()
        
        # Set custom mapping
        voice_mapper.set_custom_mapping("anime-waifu", "default_male")
        
        # Should use custom mapping
        voice = voice_mapper.get_voice_for_character("anime-waifu")
        assert voice is not None
        assert voice.name == "default_male"
        
        # Remove custom mapping
        removed = voice_mapper.remove_custom_mapping("anime-waifu")
        assert removed
        
        # Should revert to default mapping
        voice = voice_mapper.get_voice_for_character("anime-waifu")
        assert voice is not None
        assert voice.name == "anime_waifu"
    
    def test_get_available_voices_for_character(self, voice_mapper, character_manager):
        """Test getting all available voices for a character."""
        # Mock character profile for compatibility calculation
        from src.character.profile import CharacterProfile
        
        anime_profile = CharacterProfile(
            name="anime-waifu",
            description="Cute anime character",
            personality_traits=["cute", "energetic"],
            speech_patterns={},
            vocabulary_preferences={},
            transformation_prompt="Transform to anime style",
            voice_model_path="",
            intensity_multiplier=1.2
        )
        
        character_manager.load_profile.return_value = anime_profile
        voice_mapper.initialize()
        
        voices_with_scores = voice_mapper.get_available_voices_for_character("anime-waifu")
        
        assert len(voices_with_scores) > 0
        
        # Should be sorted by compatibility score
        scores = [score for _, score in voices_with_scores]
        assert scores == sorted(scores, reverse=True)
        
        # Primary voice should have highest score
        primary_voice, primary_score = voices_with_scores[0]
        assert primary_voice.name == "anime_waifu"
        assert primary_score > 0.8
    
    def test_compatibility_calculation(self, voice_mapper, character_manager):
        """Test compatibility score calculation."""
        # Mock character profile
        from src.character.profile import CharacterProfile
        
        anime_profile = CharacterProfile(
            name="anime-waifu",
            description="Cute anime character",
            personality_traits=["cute", "energetic"],
            speech_patterns={},
            vocabulary_preferences={},
            transformation_prompt="Transform to anime style",
            voice_model_path="",
            intensity_multiplier=1.2
        )
        
        character_manager.load_profile.return_value = anime_profile
        voice_mapper.initialize()
        
        # Test compatibility with female voice (should be high for cute character)
        female_voice = VoiceModel("test_female", "", 22050, "en", "female")
        compatibility = voice_mapper._calculate_compatibility("anime-waifu", female_voice)
        
        # Should have good compatibility (cute + female)
        assert compatibility > 0.7
        
        # Test compatibility with male voice (should be lower)
        male_voice = VoiceModel("test_male", "", 22050, "en", "male")
        compatibility = voice_mapper._calculate_compatibility("anime-waifu", male_voice)
        
        # Should have lower compatibility
        assert compatibility < 0.7
    
    def test_mapping_info_generation(self, voice_mapper):
        """Test mapping information generation."""
        voice_mapper.initialize()
        
        # Set a custom mapping
        voice_mapper.set_custom_mapping("anime-waifu", "default_male")
        
        info = voice_mapper.get_mapping_info()
        
        assert "total_characters" in info
        assert "custom_mappings" in info
        assert "mappings" in info
        
        assert info["total_characters"] >= 4
        assert info["custom_mappings"] >= 1
        
        # Check anime-waifu mapping info
        anime_info = info["mappings"]["anime-waifu"]
        assert anime_info["custom_override"] == "default_male"
        assert anime_info["primary_voice"] is not None


class TestVoiceModelIntegration:
    """Test integration between voice models and character profiles."""
    
    def test_character_profile_voice_model_paths(self):
        """Test that character profiles have correct voice model paths."""
        character_manager = CharacterProfileManager("characters")
        
        # Test each character profile
        characters = ["anime-waifu", "patriotic-american", "slurring-drunk"]
        
        for character_name in characters:
            profile = character_manager.load_profile(character_name)
            
            # Should have voice model path specified
            assert profile.voice_model_path, f"No voice model path for {character_name}"
            
            # Path should point to models/voices directory
            assert "models/voices" in profile.voice_model_path, f"Invalid voice model path for {character_name}"
    
    def test_voice_model_configs_exist(self):
        """Test that voice model configuration files exist."""
        models_dir = Path("models/voices")
        
        expected_configs = [
            "anime_waifu.json",
            "patriotic_american.json",
            "slurring_drunk.json",
            "default_female.json",
            "default_male.json"
        ]
        
        for config_file in expected_configs:
            config_path = models_dir / config_file
            assert config_path.exists(), f"Voice model config missing: {config_path}"
    
    def test_voice_model_configs_valid(self):
        """Test that voice model configuration files are valid."""
        models_dir = Path("models/voices")
        
        config_files = list(models_dir.glob("*.json"))
        assert len(config_files) > 0, "No voice model config files found"
        
        required_fields = ["name", "sample_rate", "language", "gender"]
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check required fields
            for field in required_fields:
                assert field in config, f"Missing field '{field}' in {config_file}"
            
            # Validate field values
            assert isinstance(config["name"], str), f"Invalid name in {config_file}"
            assert isinstance(config["sample_rate"], int), f"Invalid sample_rate in {config_file}"
            assert config["sample_rate"] > 0, f"Invalid sample_rate value in {config_file}"
            assert isinstance(config["language"], str), f"Invalid language in {config_file}"
            assert config["gender"] in ["male", "female", "neutral"], f"Invalid gender in {config_file}"
    
    def test_voice_model_character_consistency(self):
        """Test consistency between character profiles and voice model configs."""
        character_manager = CharacterProfileManager("characters")
        models_dir = Path("models/voices")
        
        # Load all character profiles
        characters = ["anime-waifu", "patriotic-american", "slurring-drunk"]
        
        for character_name in characters:
            profile = character_manager.load_profile(character_name)
            
            # Extract expected voice model name from path
            if profile.voice_model_path:
                expected_voice_name = Path(profile.voice_model_path).stem
                config_file = models_dir / f"{expected_voice_name}.json"
                
                # Config file should exist
                assert config_file.exists(), f"Voice config missing for {character_name}: {config_file}"
                
                # Load and validate config
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Name should match
                assert config["name"] == expected_voice_name, f"Name mismatch in {config_file}"


if __name__ == "__main__":
    pytest.main([__file__])