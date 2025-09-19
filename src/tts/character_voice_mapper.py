"""
Character to voice model mapping and integration utilities.

This module provides functionality to map character profiles to appropriate
voice models and handle voice model selection and fallback logic.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..core.interfaces import VoiceModel, PipelineError, PipelineStage
from ..character.profile import CharacterProfile, CharacterProfileManager
from .voice_model import VoiceModelManager

logger = logging.getLogger(__name__)


@dataclass
class VoiceMapping:
    """Represents a mapping between character and voice model."""
    character_name: str
    voice_model_name: str
    priority: int = 1  # Higher priority = preferred choice
    compatibility_score: float = 1.0  # 0.0-1.0, higher = better match


class CharacterVoiceMapper:
    """
    Maps character profiles to appropriate voice models with fallback logic.
    """
    
    def __init__(self, 
                 character_manager: CharacterProfileManager,
                 voice_manager: VoiceModelManager):
        """
        Initialize character voice mapper.
        
        Args:
            character_manager: CharacterProfileManager instance
            voice_manager: VoiceModelManager instance
        """
        self.character_manager = character_manager
        self.voice_manager = voice_manager
        self.mappings: Dict[str, List[VoiceMapping]] = {}
        self.custom_mappings: Dict[str, str] = {}
        
        # Default character to voice mappings
        self.default_mappings = {
            "default": ["default_female", "default_male"],
            "anime-waifu": ["anime_waifu", "default_female"],
            "patriotic-american": ["patriotic_american", "default_male"],
            "slurring-drunk": ["slurring_drunk", "default_male", "default_female"]
        }
    
    def initialize(self) -> None:
        """Initialize the character voice mapper."""
        try:
            # Load default mappings
            self._load_default_mappings()
            
            # Load custom mappings if they exist
            self._load_custom_mappings()
            
            # Validate all mappings
            self._validate_mappings()
            
            logger.info(f"Character voice mapper initialized with {len(self.mappings)} character mappings")
            
        except Exception as e:
            logger.error(f"Failed to initialize character voice mapper: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"Character voice mapper initialization failed: {e}",
                recoverable=False
            )
    
    def _load_default_mappings(self) -> None:
        """Load default character to voice mappings."""
        for character_name, voice_names in self.default_mappings.items():
            mappings = []
            for i, voice_name in enumerate(voice_names):
                mapping = VoiceMapping(
                    character_name=character_name,
                    voice_model_name=voice_name,
                    priority=len(voice_names) - i,  # First in list has highest priority
                    compatibility_score=1.0 if i == 0 else 0.8 - (i * 0.1)
                )
                mappings.append(mapping)
            
            self.mappings[character_name] = mappings
    
    def _load_custom_mappings(self) -> None:
        """Load custom character to voice mappings from file."""
        mappings_file = Path("config/character_voice_mappings.json")
        
        if not mappings_file.exists():
            logger.debug("No custom character voice mappings file found")
            return
        
        try:
            with open(mappings_file, 'r', encoding='utf-8') as f:
                custom_data = json.load(f)
            
            for character_name, voice_name in custom_data.items():
                self.custom_mappings[character_name] = voice_name
                logger.debug(f"Loaded custom mapping: {character_name} -> {voice_name}")
                
        except Exception as e:
            logger.warning(f"Failed to load custom character voice mappings: {e}")
    
    def _validate_mappings(self) -> None:
        """Validate that all mapped voice models exist."""
        available_voices = {voice.name for voice in self.voice_manager.get_available_voices()}
        
        for character_name, mappings in self.mappings.items():
            valid_mappings = []
            
            for mapping in mappings:
                if mapping.voice_model_name in available_voices:
                    valid_mappings.append(mapping)
                else:
                    logger.warning(f"Voice model '{mapping.voice_model_name}' not found for character '{character_name}'")
            
            if valid_mappings:
                self.mappings[character_name] = valid_mappings
            else:
                logger.error(f"No valid voice models found for character '{character_name}'")
                # Add fallback to default voices
                fallback_mappings = [
                    VoiceMapping(character_name, "default_female", 1, 0.5),
                    VoiceMapping(character_name, "default_male", 1, 0.5)
                ]
                self.mappings[character_name] = [
                    m for m in fallback_mappings 
                    if m.voice_model_name in available_voices
                ]
    
    def get_voice_for_character(self, character_name: str) -> Optional[VoiceModel]:
        """
        Get the best voice model for a character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            VoiceModel instance or None if no suitable voice found
        """
        # Check custom mappings first
        if character_name in self.custom_mappings:
            voice_name = self.custom_mappings[character_name]
            voice_model = self.voice_manager.get_voice_model(voice_name)
            if voice_model:
                logger.debug(f"Using custom voice mapping: {character_name} -> {voice_name}")
                return voice_model
        
        # Check default mappings
        if character_name in self.mappings:
            mappings = sorted(self.mappings[character_name], 
                            key=lambda m: (m.priority, m.compatibility_score), 
                            reverse=True)
            
            for mapping in mappings:
                voice_model = self.voice_manager.get_voice_model(mapping.voice_model_name)
                if voice_model:
                    logger.debug(f"Using mapped voice: {character_name} -> {mapping.voice_model_name}")
                    return voice_model
        
        # Try to get voice from character profile
        try:
            character_profile = self.character_manager.load_profile(character_name)
            if character_profile.voice_model_path:
                # Extract voice model name from path
                voice_name = Path(character_profile.voice_model_path).stem
                voice_model = self.voice_manager.get_voice_model(voice_name)
                if voice_model:
                    logger.debug(f"Using profile voice: {character_name} -> {voice_name}")
                    return voice_model
        except Exception as e:
            logger.warning(f"Failed to load character profile for {character_name}: {e}")
        
        # Fallback to default voice
        logger.warning(f"No specific voice found for character '{character_name}', using fallback")
        return self.voice_manager.get_fallback_voice()
    
    def get_available_voices_for_character(self, character_name: str) -> List[Tuple[VoiceModel, float]]:
        """
        Get all available voice models for a character with compatibility scores.
        
        Args:
            character_name: Name of the character
            
        Returns:
            List of (VoiceModel, compatibility_score) tuples, sorted by compatibility
        """
        results = []
        
        # Get mapped voices
        if character_name in self.mappings:
            for mapping in self.mappings[character_name]:
                voice_model = self.voice_manager.get_voice_model(mapping.voice_model_name)
                if voice_model:
                    results.append((voice_model, mapping.compatibility_score))
        
        # Add other available voices with lower compatibility
        mapped_voice_names = {mapping.voice_model_name for mapping in self.mappings.get(character_name, [])}
        
        for voice_model in self.voice_manager.get_available_voices():
            if voice_model.name not in mapped_voice_names:
                # Calculate compatibility based on character and voice characteristics
                compatibility = self._calculate_compatibility(character_name, voice_model)
                results.append((voice_model, compatibility))
        
        # Sort by compatibility score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculate_compatibility(self, character_name: str, voice_model: VoiceModel) -> float:
        """
        Calculate compatibility score between character and voice model.
        
        Args:
            character_name: Name of the character
            voice_model: VoiceModel to evaluate
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        try:
            character_profile = self.character_manager.load_profile(character_name)
        except Exception:
            return 0.3  # Default low compatibility
        
        score = 0.3  # Base score
        
        # Language compatibility
        if voice_model.language == "en":  # All our characters speak English
            score += 0.2
        
        # Gender compatibility based on character traits
        if hasattr(character_profile, 'personality_traits') and character_profile.personality_traits:
            personality_str = " ".join(character_profile.personality_traits).lower()
            
            if "cute" in personality_str or "kawaii" in personality_str:
                if voice_model.gender == "female":
                    score += 0.3
            elif "patriotic" in personality_str or "confident" in personality_str:
                if voice_model.gender == "male":
                    score += 0.3
            elif "drunk" in personality_str or "relaxed" in personality_str:
                score += 0.2  # Neutral gender preference
        
        # Name-based compatibility
        if character_name.replace("-", "_") in voice_model.name:
            score += 0.2
        
        return min(score, 1.0)
    
    def set_custom_mapping(self, character_name: str, voice_model_name: str) -> None:
        """
        Set a custom character to voice mapping.
        
        Args:
            character_name: Name of the character
            voice_model_name: Name of the voice model
        """
        # Validate that voice model exists
        voice_model = self.voice_manager.get_voice_model(voice_model_name)
        if not voice_model:
            raise ValueError(f"Voice model '{voice_model_name}' not found")
        
        self.custom_mappings[character_name] = voice_model_name
        logger.info(f"Set custom mapping: {character_name} -> {voice_model_name}")
    
    def remove_custom_mapping(self, character_name: str) -> bool:
        """
        Remove a custom character to voice mapping.
        
        Args:
            character_name: Name of the character
            
        Returns:
            True if mapping was removed, False if it didn't exist
        """
        if character_name in self.custom_mappings:
            del self.custom_mappings[character_name]
            logger.info(f"Removed custom mapping for character: {character_name}")
            return True
        return False
    
    def save_custom_mappings(self) -> None:
        """Save custom character to voice mappings to file."""
        mappings_file = Path("config/character_voice_mappings.json")
        mappings_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_mappings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved custom character voice mappings to {mappings_file}")
            
        except Exception as e:
            logger.error(f"Failed to save custom character voice mappings: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"Failed to save custom mappings: {e}",
                recoverable=True
            )
    
    def get_mapping_info(self) -> Dict[str, Any]:
        """
        Get information about character to voice mappings.
        
        Returns:
            Dictionary with mapping information
        """
        info = {
            "total_characters": len(self.mappings),
            "custom_mappings": len(self.custom_mappings),
            "mappings": {}
        }
        
        for character_name, mappings in self.mappings.items():
            character_info = {
                "available_voices": len(mappings),
                "primary_voice": mappings[0].voice_model_name if mappings else None,
                "fallback_voices": [m.voice_model_name for m in mappings[1:]] if len(mappings) > 1 else [],
                "custom_override": self.custom_mappings.get(character_name)
            }
            info["mappings"][character_name] = character_info
        
        return info