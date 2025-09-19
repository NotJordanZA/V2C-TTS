"""
Character profile management system for voice transformation.

This module provides the CharacterProfile dataclass and utilities for loading,
validating, and managing character profiles from JSON configuration files.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CharacterProfile:
    """
    Represents a character profile with all attributes needed for voice transformation.
    
    Attributes:
        name: Unique identifier for the character
        description: Human-readable description of the character
        personality_traits: List of personality characteristics
        speech_patterns: Dictionary mapping patterns to replacements
        vocabulary_preferences: Dictionary of preferred words/phrases by category
        transformation_prompt: LLM prompt template for character transformation
        voice_model_path: Path to the voice model file for TTS
        intensity_multiplier: Multiplier for transformation intensity (0.0-2.0)
    """
    name: str
    description: str
    personality_traits: List[str] = field(default_factory=list)
    speech_patterns: Dict[str, str] = field(default_factory=dict)
    vocabulary_preferences: Dict[str, List[str]] = field(default_factory=dict)
    transformation_prompt: str = ""
    voice_model_path: str = ""
    intensity_multiplier: float = 1.0
    
    def __post_init__(self):
        """Validate character profile attributes after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate character profile attributes."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Character name must be a non-empty string")
        
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Character description must be a non-empty string")
        
        if not isinstance(self.personality_traits, list):
            raise ValueError("Personality traits must be a list")
        
        if not isinstance(self.speech_patterns, dict):
            raise ValueError("Speech patterns must be a dictionary")
        
        if not isinstance(self.vocabulary_preferences, dict):
            raise ValueError("Vocabulary preferences must be a dictionary")
        
        if not isinstance(self.transformation_prompt, str):
            raise ValueError("Transformation prompt must be a string")
        
        if not isinstance(self.voice_model_path, str):
            raise ValueError("Voice model path must be a string")
        
        if not isinstance(self.intensity_multiplier, (int, float)):
            raise ValueError("Intensity multiplier must be a number")
        
        if not (0.0 <= self.intensity_multiplier <= 2.0):
            raise ValueError("Intensity multiplier must be between 0.0 and 2.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert character profile to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterProfile':
        """Create character profile from dictionary."""
        return cls(**data)


class CharacterProfileManager:
    """
    Manages character profiles including loading, validation, and caching.
    """
    
    def __init__(self, profiles_directory: str = "characters"):
        """
        Initialize character profile manager.
        
        Args:
            profiles_directory: Directory containing character profile JSON files
        """
        self.profiles_directory = Path(profiles_directory)
        self._profiles_cache: Dict[str, CharacterProfile] = {}
        self._default_profile: Optional[CharacterProfile] = None
        
        # Ensure profiles directory exists
        self.profiles_directory.mkdir(exist_ok=True)
        
        # Load default profile
        self._load_default_profile()
    
    def _load_default_profile(self):
        """Load or create default character profile."""
        try:
            self._default_profile = self.load_profile("default")
        except FileNotFoundError:
            # Create default profile if it doesn't exist
            self._default_profile = CharacterProfile(
                name="default",
                description="Default neutral character with minimal transformations",
                personality_traits=["neutral", "clear"],
                speech_patterns={},
                vocabulary_preferences={},
                transformation_prompt="Repeat the following text with minimal changes: {text}",
                voice_model_path="",
                intensity_multiplier=1.0
            )
            self.save_profile(self._default_profile)
    
    def load_profile(self, character_name: str) -> CharacterProfile:
        """
        Load character profile from JSON file.
        
        Args:
            character_name: Name of the character profile to load
            
        Returns:
            CharacterProfile instance
            
        Raises:
            FileNotFoundError: If profile file doesn't exist
            ValueError: If profile data is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        # Check cache first
        if character_name in self._profiles_cache:
            return self._profiles_cache[character_name]
        
        profile_path = self.profiles_directory / f"{character_name}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Character profile not found: {profile_path}")
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            profile = CharacterProfile.from_dict(profile_data)
            
            # Cache the loaded profile
            self._profiles_cache[character_name] = profile
            
            logger.info(f"Loaded character profile: {character_name}")
            return profile
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in profile {character_name}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load character profile {character_name}: {e}")
    
    def save_profile(self, profile: CharacterProfile) -> None:
        """
        Save character profile to JSON file.
        
        Args:
            profile: CharacterProfile instance to save
            
        Raises:
            OSError: If file cannot be written
        """
        profile_path = self.profiles_directory / f"{profile.name}.json"
        
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Update cache
            self._profiles_cache[profile.name] = profile
            
            logger.info(f"Saved character profile: {profile.name}")
            
        except Exception as e:
            raise OSError(f"Failed to save character profile {profile.name}: {e}")
    
    def get_available_profiles(self) -> List[str]:
        """
        Get list of available character profile names.
        
        Returns:
            List of character profile names
        """
        profile_files = list(self.profiles_directory.glob("*.json"))
        return [f.stem for f in profile_files]
    
    def get_default_profile(self) -> CharacterProfile:
        """
        Get the default character profile.
        
        Returns:
            Default CharacterProfile instance
        """
        return self._default_profile
    
    def validate_profile(self, profile: CharacterProfile) -> bool:
        """
        Validate character profile.
        
        Args:
            profile: CharacterProfile to validate
            
        Returns:
            True if profile is valid
            
        Raises:
            ValueError: If profile is invalid
        """
        try:
            profile._validate()
            return True
        except ValueError:
            raise
    
    def delete_profile(self, character_name: str) -> None:
        """
        Delete character profile.
        
        Args:
            character_name: Name of character profile to delete
            
        Raises:
            FileNotFoundError: If profile doesn't exist
            ValueError: If trying to delete default profile
        """
        if character_name == "default":
            raise ValueError("Cannot delete default profile")
        
        profile_path = self.profiles_directory / f"{character_name}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Character profile not found: {character_name}")
        
        profile_path.unlink()
        
        # Remove from cache
        if character_name in self._profiles_cache:
            del self._profiles_cache[character_name]
        
        logger.info(f"Deleted character profile: {character_name}")
    
    def clear_cache(self) -> None:
        """Clear the profiles cache."""
        self._profiles_cache.clear()
        logger.info("Cleared character profiles cache")