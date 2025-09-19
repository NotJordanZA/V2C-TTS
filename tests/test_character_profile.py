"""
Unit tests for character profile management system.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.character.profile import CharacterProfile, CharacterProfileManager


class TestCharacterProfile:
    """Test cases for CharacterProfile dataclass."""
    
    def test_character_profile_creation(self):
        """Test creating a valid character profile."""
        profile = CharacterProfile(
            name="test_character",
            description="A test character",
            personality_traits=["friendly", "helpful"],
            speech_patterns={"hello": "hi there"},
            vocabulary_preferences={"greetings": ["hello", "hi"]},
            transformation_prompt="Transform: {text}",
            voice_model_path="/path/to/model.pth",
            intensity_multiplier=1.0
        )
        
        assert profile.name == "test_character"
        assert profile.description == "A test character"
        assert profile.personality_traits == ["friendly", "helpful"]
        assert profile.speech_patterns == {"hello": "hi there"}
        assert profile.vocabulary_preferences == {"greetings": ["hello", "hi"]}
        assert profile.transformation_prompt == "Transform: {text}"
        assert profile.voice_model_path == "/path/to/model.pth"
        assert profile.intensity_multiplier == 1.0
    
    def test_character_profile_defaults(self):
        """Test character profile with default values."""
        profile = CharacterProfile(
            name="minimal",
            description="Minimal character"
        )
        
        assert profile.personality_traits == []
        assert profile.speech_patterns == {}
        assert profile.vocabulary_preferences == {}
        assert profile.transformation_prompt == ""
        assert profile.voice_model_path == ""
        assert profile.intensity_multiplier == 1.0
    
    def test_character_profile_validation_empty_name(self):
        """Test validation fails with empty name."""
        with pytest.raises(ValueError, match="Character name must be a non-empty string"):
            CharacterProfile(name="", description="Test")
    
    def test_character_profile_validation_none_name(self):
        """Test validation fails with None name."""
        with pytest.raises(ValueError, match="Character name must be a non-empty string"):
            CharacterProfile(name=None, description="Test")
    
    def test_character_profile_validation_empty_description(self):
        """Test validation fails with empty description."""
        with pytest.raises(ValueError, match="Character description must be a non-empty string"):
            CharacterProfile(name="test", description="")
    
    def test_character_profile_validation_invalid_personality_traits(self):
        """Test validation fails with invalid personality traits."""
        with pytest.raises(ValueError, match="Personality traits must be a list"):
            CharacterProfile(
                name="test",
                description="Test",
                personality_traits="not a list"
            )
    
    def test_character_profile_validation_invalid_speech_patterns(self):
        """Test validation fails with invalid speech patterns."""
        with pytest.raises(ValueError, match="Speech patterns must be a dictionary"):
            CharacterProfile(
                name="test",
                description="Test",
                speech_patterns="not a dict"
            )
    
    def test_character_profile_validation_invalid_vocabulary_preferences(self):
        """Test validation fails with invalid vocabulary preferences."""
        with pytest.raises(ValueError, match="Vocabulary preferences must be a dictionary"):
            CharacterProfile(
                name="test",
                description="Test",
                vocabulary_preferences="not a dict"
            )
    
    def test_character_profile_validation_invalid_transformation_prompt(self):
        """Test validation fails with invalid transformation prompt."""
        with pytest.raises(ValueError, match="Transformation prompt must be a string"):
            CharacterProfile(
                name="test",
                description="Test",
                transformation_prompt=123
            )
    
    def test_character_profile_validation_invalid_voice_model_path(self):
        """Test validation fails with invalid voice model path."""
        with pytest.raises(ValueError, match="Voice model path must be a string"):
            CharacterProfile(
                name="test",
                description="Test",
                voice_model_path=123
            )
    
    def test_character_profile_validation_invalid_intensity_multiplier_type(self):
        """Test validation fails with invalid intensity multiplier type."""
        with pytest.raises(ValueError, match="Intensity multiplier must be a number"):
            CharacterProfile(
                name="test",
                description="Test",
                intensity_multiplier="not a number"
            )
    
    def test_character_profile_validation_intensity_multiplier_out_of_range_low(self):
        """Test validation fails with intensity multiplier below range."""
        with pytest.raises(ValueError, match="Intensity multiplier must be between 0.0 and 2.0"):
            CharacterProfile(
                name="test",
                description="Test",
                intensity_multiplier=-0.1
            )
    
    def test_character_profile_validation_intensity_multiplier_out_of_range_high(self):
        """Test validation fails with intensity multiplier above range."""
        with pytest.raises(ValueError, match="Intensity multiplier must be between 0.0 and 2.0"):
            CharacterProfile(
                name="test",
                description="Test",
                intensity_multiplier=2.1
            )
    
    def test_character_profile_to_dict(self):
        """Test converting character profile to dictionary."""
        profile = CharacterProfile(
            name="test",
            description="Test character",
            personality_traits=["trait1"],
            speech_patterns={"a": "b"},
            vocabulary_preferences={"cat": ["words"]},
            transformation_prompt="prompt",
            voice_model_path="path",
            intensity_multiplier=1.5
        )
        
        expected_dict = {
            "name": "test",
            "description": "Test character",
            "personality_traits": ["trait1"],
            "speech_patterns": {"a": "b"},
            "vocabulary_preferences": {"cat": ["words"]},
            "transformation_prompt": "prompt",
            "voice_model_path": "path",
            "intensity_multiplier": 1.5
        }
        
        assert profile.to_dict() == expected_dict
    
    def test_character_profile_from_dict(self):
        """Test creating character profile from dictionary."""
        data = {
            "name": "test",
            "description": "Test character",
            "personality_traits": ["trait1"],
            "speech_patterns": {"a": "b"},
            "vocabulary_preferences": {"cat": ["words"]},
            "transformation_prompt": "prompt",
            "voice_model_path": "path",
            "intensity_multiplier": 1.5
        }
        
        profile = CharacterProfile.from_dict(data)
        
        assert profile.name == "test"
        assert profile.description == "Test character"
        assert profile.personality_traits == ["trait1"]
        assert profile.speech_patterns == {"a": "b"}
        assert profile.vocabulary_preferences == {"cat": ["words"]}
        assert profile.transformation_prompt == "prompt"
        assert profile.voice_model_path == "path"
        assert profile.intensity_multiplier == 1.5


class TestCharacterProfileManager:
    """Test cases for CharacterProfileManager."""
    
    @pytest.fixture
    def temp_profiles_dir(self):
        """Create temporary directory for test profiles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_profile_data(self):
        """Sample character profile data for testing."""
        return {
            "name": "test_character",
            "description": "A test character",
            "personality_traits": ["friendly"],
            "speech_patterns": {"hello": "hi"},
            "vocabulary_preferences": {"greetings": ["hello"]},
            "transformation_prompt": "Transform: {text}",
            "voice_model_path": "/path/to/model.pth",
            "intensity_multiplier": 1.0
        }
    
    def test_character_profile_manager_initialization(self, temp_profiles_dir):
        """Test CharacterProfileManager initialization."""
        manager = CharacterProfileManager(temp_profiles_dir)
        
        assert manager.profiles_directory == Path(temp_profiles_dir)
        assert manager.profiles_directory.exists()
        assert manager._default_profile is not None
        assert manager._default_profile.name == "default"
    
    def test_save_and_load_profile(self, temp_profiles_dir, sample_profile_data):
        """Test saving and loading character profiles."""
        manager = CharacterProfileManager(temp_profiles_dir)
        profile = CharacterProfile.from_dict(sample_profile_data)
        
        # Save profile
        manager.save_profile(profile)
        
        # Verify file was created
        profile_path = Path(temp_profiles_dir) / "test_character.json"
        assert profile_path.exists()
        
        # Load profile
        loaded_profile = manager.load_profile("test_character")
        
        assert loaded_profile.name == profile.name
        assert loaded_profile.description == profile.description
        assert loaded_profile.personality_traits == profile.personality_traits
        assert loaded_profile.speech_patterns == profile.speech_patterns
        assert loaded_profile.vocabulary_preferences == profile.vocabulary_preferences
        assert loaded_profile.transformation_prompt == profile.transformation_prompt
        assert loaded_profile.voice_model_path == profile.voice_model_path
        assert loaded_profile.intensity_multiplier == profile.intensity_multiplier
    
    def test_load_nonexistent_profile(self, temp_profiles_dir):
        """Test loading a profile that doesn't exist."""
        manager = CharacterProfileManager(temp_profiles_dir)
        
        with pytest.raises(FileNotFoundError, match="Character profile not found"):
            manager.load_profile("nonexistent")
    
    def test_load_invalid_json_profile(self, temp_profiles_dir):
        """Test loading a profile with invalid JSON."""
        manager = CharacterProfileManager(temp_profiles_dir)
        
        # Create invalid JSON file
        invalid_profile_path = Path(temp_profiles_dir) / "invalid.json"
        with open(invalid_profile_path, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(ValueError, match="Invalid JSON in profile"):
            manager.load_profile("invalid")
    
    def test_load_invalid_profile_data(self, temp_profiles_dir):
        """Test loading a profile with invalid data."""
        manager = CharacterProfileManager(temp_profiles_dir)
        
        # Create profile with invalid data
        invalid_data = {"name": "", "description": ""}  # Empty name should fail validation
        invalid_profile_path = Path(temp_profiles_dir) / "invalid_data.json"
        with open(invalid_profile_path, 'w') as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError):
            manager.load_profile("invalid_data")
    
    def test_get_available_profiles(self, temp_profiles_dir, sample_profile_data):
        """Test getting list of available profiles."""
        manager = CharacterProfileManager(temp_profiles_dir)
        
        # Initially should have default profile
        profiles = manager.get_available_profiles()
        assert "default" in profiles
        
        # Add another profile
        profile = CharacterProfile.from_dict(sample_profile_data)
        manager.save_profile(profile)
        
        profiles = manager.get_available_profiles()
        assert "default" in profiles
        assert "test_character" in profiles
        assert len(profiles) == 2
    
    def test_get_default_profile(self, temp_profiles_dir):
        """Test getting default profile."""
        manager = CharacterProfileManager(temp_profiles_dir)
        default_profile = manager.get_default_profile()
        
        assert default_profile.name == "default"
        assert default_profile.description == "Default neutral character with minimal transformations"
        assert default_profile.intensity_multiplier == 1.0
    
    def test_validate_profile(self, temp_profiles_dir):
        """Test profile validation."""
        manager = CharacterProfileManager(temp_profiles_dir)
        
        # Valid profile
        valid_profile = CharacterProfile(name="valid", description="Valid profile")
        assert manager.validate_profile(valid_profile) is True
        
        # Invalid profile
        with pytest.raises(ValueError):
            invalid_profile = CharacterProfile(name="", description="Invalid")
            manager.validate_profile(invalid_profile)
    
    def test_delete_profile(self, temp_profiles_dir, sample_profile_data):
        """Test deleting character profiles."""
        manager = CharacterProfileManager(temp_profiles_dir)
        profile = CharacterProfile.from_dict(sample_profile_data)
        
        # Save profile first
        manager.save_profile(profile)
        assert "test_character" in manager.get_available_profiles()
        
        # Delete profile
        manager.delete_profile("test_character")
        assert "test_character" not in manager.get_available_profiles()
        
        # Verify file was deleted
        profile_path = Path(temp_profiles_dir) / "test_character.json"
        assert not profile_path.exists()
    
    def test_delete_nonexistent_profile(self, temp_profiles_dir):
        """Test deleting a profile that doesn't exist."""
        manager = CharacterProfileManager(temp_profiles_dir)
        
        with pytest.raises(FileNotFoundError, match="Character profile not found"):
            manager.delete_profile("nonexistent")
    
    def test_delete_default_profile(self, temp_profiles_dir):
        """Test that default profile cannot be deleted."""
        manager = CharacterProfileManager(temp_profiles_dir)
        
        with pytest.raises(ValueError, match="Cannot delete default profile"):
            manager.delete_profile("default")
    
    def test_profile_caching(self, temp_profiles_dir, sample_profile_data):
        """Test that profiles are cached after loading."""
        manager = CharacterProfileManager(temp_profiles_dir)
        profile = CharacterProfile.from_dict(sample_profile_data)
        manager.save_profile(profile)
        
        # Load profile first time
        profile1 = manager.load_profile("test_character")
        
        # Load profile second time (should come from cache)
        profile2 = manager.load_profile("test_character")
        
        # Should be the same object (cached)
        assert profile1 is profile2
    
    def test_clear_cache(self, temp_profiles_dir, sample_profile_data):
        """Test clearing the profiles cache."""
        manager = CharacterProfileManager(temp_profiles_dir)
        profile = CharacterProfile.from_dict(sample_profile_data)
        manager.save_profile(profile)
        
        # Load profile to cache it
        profile1 = manager.load_profile("test_character")
        
        # Clear cache
        manager.clear_cache()
        
        # Load profile again (should be new object)
        profile2 = manager.load_profile("test_character")
        
        # Should be different objects
        assert profile1 is not profile2
        # But should have same data
        assert profile1.name == profile2.name