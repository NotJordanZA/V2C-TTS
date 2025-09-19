"""
Tests for character profile validation and loading.

This module tests the validation of all default character profiles to ensure
they conform to the CharacterProfile schema and contain appropriate data.
"""

import pytest
import json
from pathlib import Path
from src.character.profile import CharacterProfile, CharacterProfileManager


class TestCharacterProfileValidation:
    """Test validation of all default character profiles."""
    
    @pytest.fixture
    def profile_manager(self):
        """Create character profile manager for testing."""
        return CharacterProfileManager("characters")
    
    @pytest.fixture
    def character_names(self):
        """List of all default character names to test."""
        return ["default", "anime-waifu", "patriotic-american", "slurring-drunk"]
    
    def test_all_profiles_exist(self, character_names):
        """Test that all expected character profile files exist."""
        characters_dir = Path("characters")
        
        for character_name in character_names:
            profile_path = characters_dir / f"{character_name}.json"
            assert profile_path.exists(), f"Character profile file missing: {profile_path}"
    
    def test_all_profiles_valid_json(self, character_names):
        """Test that all character profile files contain valid JSON."""
        characters_dir = Path("characters")
        
        for character_name in character_names:
            profile_path = characters_dir / f"{character_name}.json"
            
            with open(profile_path, 'r', encoding='utf-8') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {character_name}.json: {e}")
    
    def test_all_profiles_load_successfully(self, profile_manager, character_names):
        """Test that all character profiles can be loaded without errors."""
        for character_name in character_names:
            try:
                profile = profile_manager.load_profile(character_name)
                assert isinstance(profile, CharacterProfile)
                assert profile.name == character_name
            except Exception as e:
                pytest.fail(f"Failed to load character profile {character_name}: {e}")
    
    def test_all_profiles_have_required_fields(self, profile_manager, character_names):
        """Test that all character profiles have required fields."""
        required_fields = [
            "name", "description", "personality_traits", "speech_patterns",
            "vocabulary_preferences", "transformation_prompt", "voice_model_path",
            "intensity_multiplier"
        ]
        
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            profile_dict = profile.to_dict()
            
            for field in required_fields:
                assert field in profile_dict, f"Missing field '{field}' in {character_name}"
                assert profile_dict[field] is not None, f"Field '{field}' is None in {character_name}"
    
    def test_profile_name_consistency(self, profile_manager, character_names):
        """Test that profile names match their file names."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            assert profile.name == character_name, f"Profile name mismatch in {character_name}"
    
    def test_profile_descriptions_not_empty(self, profile_manager, character_names):
        """Test that all profiles have non-empty descriptions."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            assert profile.description.strip(), f"Empty description in {character_name}"
            assert len(profile.description) > 10, f"Description too short in {character_name}"
    
    def test_personality_traits_valid(self, profile_manager, character_names):
        """Test that personality traits are valid lists with content."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            
            assert isinstance(profile.personality_traits, list), f"Personality traits not a list in {character_name}"
            assert len(profile.personality_traits) > 0, f"No personality traits in {character_name}"
            
            for trait in profile.personality_traits:
                assert isinstance(trait, str), f"Non-string personality trait in {character_name}"
                assert trait.strip(), f"Empty personality trait in {character_name}"
    
    def test_speech_patterns_valid(self, profile_manager, character_names):
        """Test that speech patterns are valid dictionaries."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            
            assert isinstance(profile.speech_patterns, dict), f"Speech patterns not a dict in {character_name}"
            
            for key, value in profile.speech_patterns.items():
                assert isinstance(key, str), f"Non-string speech pattern key in {character_name}"
                assert isinstance(value, str), f"Non-string speech pattern value in {character_name}"
                assert key.strip(), f"Empty speech pattern key in {character_name}"
                assert value.strip(), f"Empty speech pattern value in {character_name}"
    
    def test_vocabulary_preferences_valid(self, profile_manager, character_names):
        """Test that vocabulary preferences are valid dictionaries."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            
            assert isinstance(profile.vocabulary_preferences, dict), f"Vocabulary preferences not a dict in {character_name}"
            
            for category, words in profile.vocabulary_preferences.items():
                assert isinstance(category, str), f"Non-string vocabulary category in {character_name}"
                assert isinstance(words, list), f"Non-list vocabulary words in {character_name}"
                assert category.strip(), f"Empty vocabulary category in {character_name}"
                
                for word in words:
                    assert isinstance(word, str), f"Non-string vocabulary word in {character_name}"
                    assert word.strip(), f"Empty vocabulary word in {character_name}"
    
    def test_transformation_prompt_valid(self, profile_manager, character_names):
        """Test that transformation prompts are valid and contain placeholder."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            
            assert isinstance(profile.transformation_prompt, str), f"Transformation prompt not a string in {character_name}"
            assert profile.transformation_prompt.strip(), f"Empty transformation prompt in {character_name}"
            assert "{text}" in profile.transformation_prompt, f"Missing {{text}} placeholder in {character_name}"
            assert len(profile.transformation_prompt) > 20, f"Transformation prompt too short in {character_name}"
    
    def test_voice_model_path_valid(self, profile_manager, character_names):
        """Test that voice model paths are valid strings."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            
            assert isinstance(profile.voice_model_path, str), f"Voice model path not a string in {character_name}"
            # Note: We don't require the path to exist since voice models may not be downloaded yet
    
    def test_intensity_multiplier_valid(self, profile_manager, character_names):
        """Test that intensity multipliers are valid numbers in range."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            
            assert isinstance(profile.intensity_multiplier, (int, float)), f"Intensity multiplier not a number in {character_name}"
            assert 0.0 <= profile.intensity_multiplier <= 2.0, f"Intensity multiplier out of range in {character_name}"
    
    def test_profile_validation_passes(self, profile_manager, character_names):
        """Test that all profiles pass internal validation."""
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            
            try:
                profile._validate()
            except ValueError as e:
                pytest.fail(f"Profile validation failed for {character_name}: {e}")


class TestSpecificCharacterProfiles:
    """Test specific characteristics of individual character profiles."""
    
    @pytest.fixture
    def profile_manager(self):
        """Create character profile manager for testing."""
        return CharacterProfileManager("characters")
    
    def test_default_profile_characteristics(self, profile_manager):
        """Test that default profile has neutral characteristics."""
        profile = profile_manager.load_profile("default")
        
        assert "neutral" in profile.personality_traits
        assert len(profile.speech_patterns) == 0  # Should have no speech pattern modifications
        assert len(profile.vocabulary_preferences) == 0  # Should have no vocabulary preferences
        assert profile.intensity_multiplier == 1.0  # Should be neutral intensity
        assert "minimal changes" in profile.transformation_prompt.lower()
    
    def test_anime_waifu_profile_characteristics(self, profile_manager):
        """Test that anime waifu profile has appropriate characteristics."""
        profile = profile_manager.load_profile("anime-waifu")
        
        # Should have cute/kawaii traits
        cute_traits = ["cute", "kawaii", "adorable", "sweet", "innocent"]
        assert any(trait in profile.personality_traits for trait in cute_traits)
        
        # Should have Japanese expressions in speech patterns or vocabulary
        japanese_elements = ["hai", "arigatou", "sayonara", "konnichiwa", "desu", "nya"]
        has_japanese = (
            any(element in str(profile.speech_patterns).lower() for element in japanese_elements) or
            any(element in str(profile.vocabulary_preferences).lower() for element in japanese_elements)
        )
        assert has_japanese, "Anime waifu profile should contain Japanese expressions"
        
        # Should have higher intensity for more transformation
        assert profile.intensity_multiplier > 1.0
        
        # Transformation prompt should mention anime/kawaii characteristics
        prompt_lower = profile.transformation_prompt.lower()
        anime_keywords = ["anime", "kawaii", "cute", "adorable"]
        assert any(keyword in prompt_lower for keyword in anime_keywords)
    
    def test_patriotic_american_profile_characteristics(self, profile_manager):
        """Test that patriotic American profile has appropriate characteristics."""
        profile = profile_manager.load_profile("patriotic-american")
        
        # Should have patriotic traits
        patriotic_traits = ["patriotic", "proud", "confident", "enthusiastic"]
        assert any(trait in profile.personality_traits for trait in patriotic_traits)
        
        # Should have American/patriotic vocabulary
        american_elements = ["america", "freedom", "liberty", "usa", "constitution"]
        has_american = (
            any(element in str(profile.speech_patterns).lower() for element in american_elements) or
            any(element in str(profile.vocabulary_preferences).lower() for element in american_elements)
        )
        assert has_american, "Patriotic American profile should contain patriotic vocabulary"
        
        # Should have higher intensity for strong transformation
        assert profile.intensity_multiplier > 1.0
        
        # Transformation prompt should mention patriotic characteristics
        prompt_lower = profile.transformation_prompt.lower()
        patriotic_keywords = ["patriotic", "american", "freedom", "liberty", "proud"]
        assert any(keyword in prompt_lower for keyword in patriotic_keywords)
    
    def test_slurring_drunk_profile_characteristics(self, profile_manager):
        """Test that slurring drunk profile has appropriate characteristics."""
        profile = profile_manager.load_profile("slurring-drunk")
        
        # Should have drunk-related traits
        drunk_traits = ["relaxed", "uninhibited", "confused", "rambling", "emotional"]
        assert any(trait in profile.personality_traits for trait in drunk_traits)
        
        # Should have speech pattern modifications for slurring
        speech_modifications = ["sh", "da", "dish", "dat", "ya", "yer"]
        has_slurring = any(mod in str(profile.speech_patterns).lower() for mod in speech_modifications)
        assert has_slurring, "Slurring drunk profile should contain speech modifications"
        
        # Should have filler words in vocabulary
        filler_words = ["uh", "uhm", "ya know", "like", "man", "dude"]
        has_fillers = any(filler in str(profile.vocabulary_preferences).lower() for filler in filler_words)
        assert has_fillers, "Slurring drunk profile should contain filler words"
        
        # Should have higher intensity for noticeable transformation
        assert profile.intensity_multiplier > 1.0
        
        # Transformation prompt should mention slurring/drunk characteristics
        prompt_lower = profile.transformation_prompt.lower()
        drunk_keywords = ["slur", "drunk", "drinking", "relaxed", "filler"]
        assert any(keyword in prompt_lower for keyword in drunk_keywords)


class TestCharacterProfileManager:
    """Test the CharacterProfileManager functionality with default profiles."""
    
    @pytest.fixture
    def profile_manager(self):
        """Create character profile manager for testing."""
        return CharacterProfileManager("characters")
    
    def test_get_available_profiles_includes_defaults(self, profile_manager):
        """Test that get_available_profiles returns all default profiles."""
        available_profiles = profile_manager.get_available_profiles()
        expected_profiles = ["default", "anime-waifu", "patriotic-american", "slurring-drunk"]
        
        for expected_profile in expected_profiles:
            assert expected_profile in available_profiles, f"Missing profile: {expected_profile}"
    
    def test_default_profile_accessible(self, profile_manager):
        """Test that default profile is accessible via get_default_profile."""
        default_profile = profile_manager.get_default_profile()
        
        assert isinstance(default_profile, CharacterProfile)
        assert default_profile.name == "default"
    
    def test_profile_caching_works(self, profile_manager):
        """Test that profile caching works correctly."""
        # Load profile twice
        profile1 = profile_manager.load_profile("anime-waifu")
        profile2 = profile_manager.load_profile("anime-waifu")
        
        # Should be the same object (cached)
        assert profile1 is profile2
    
    def test_profile_validation_method(self, profile_manager):
        """Test that profile validation method works for all profiles."""
        character_names = ["default", "anime-waifu", "patriotic-american", "slurring-drunk"]
        
        for character_name in character_names:
            profile = profile_manager.load_profile(character_name)
            
            try:
                is_valid = profile_manager.validate_profile(profile)
                assert is_valid, f"Profile validation failed for {character_name}"
            except ValueError as e:
                pytest.fail(f"Profile validation raised error for {character_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])