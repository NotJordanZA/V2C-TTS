#!/usr/bin/env python3
"""
Voice Model Integration Demo

This script demonstrates the voice model integration system, showing how
character profiles are mapped to voice models and how the system handles
voice model loading and fallback logic.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.character.profile import CharacterProfileManager
from src.tts.voice_model import VoiceModelManager
from src.tts.character_voice_mapper import CharacterVoiceMapper


def main():
    """Demonstrate voice model integration."""
    print("=== Voice Model Integration Demo ===\n")
    
    try:
        # Initialize managers
        print("1. Initializing managers...")
        character_manager = CharacterProfileManager("characters")
        voice_manager = VoiceModelManager("models/voices")
        voice_mapper = CharacterVoiceMapper(character_manager, voice_manager)
        
        # Initialize voice manager and mapper
        voice_manager.initialize()
        voice_mapper.initialize()
        
        print(f"   ✓ Loaded {len(voice_manager.get_available_voices())} voice models")
        print(f"   ✓ Configured mappings for {len(voice_mapper.mappings)} characters\n")
        
        # Show available voice models
        print("2. Available Voice Models:")
        voices = voice_manager.get_available_voices()
        for voice in voices:
            print(f"   • {voice.name} ({voice.language}, {voice.gender})")
            if voice.model_path:
                print(f"     Path: {voice.model_path}")
            else:
                print(f"     Built-in voice")
        print()
        
        # Show character to voice mappings
        print("3. Character to Voice Mappings:")
        characters = ["default", "anime-waifu", "patriotic-american", "slurring-drunk"]
        
        for character_name in characters:
            print(f"\n   Character: {character_name}")
            
            # Get primary voice
            primary_voice = voice_mapper.get_voice_for_character(character_name)
            if primary_voice:
                print(f"   Primary Voice: {primary_voice.name} ({primary_voice.gender})")
            else:
                print(f"   Primary Voice: None (fallback will be used)")
            
            # Get all available voices with compatibility scores
            available_voices = voice_mapper.get_available_voices_for_character(character_name)
            print(f"   Available Voices ({len(available_voices)}):")
            
            for voice, score in available_voices[:3]:  # Show top 3
                print(f"     • {voice.name} (compatibility: {score:.2f})")
        
        print("\n4. Voice Model Information:")
        model_info = voice_manager.get_model_info()
        print(f"   Total Models: {model_info['total_models']}")
        print(f"   Models Directory: {model_info['models_directory']}")
        print(f"   Supported Formats: {', '.join(model_info['supported_formats'])}")
        
        print("\n   By Language:")
        for language, voice_names in model_info['by_language'].items():
            print(f"     {language}: {', '.join(voice_names)}")
        
        print("\n   By Gender:")
        for gender, voice_names in model_info['by_gender'].items():
            print(f"     {gender}: {', '.join(voice_names)}")
        
        print("\n5. Character Voice Mapping Information:")
        mapping_info = voice_mapper.get_mapping_info()
        print(f"   Total Characters: {mapping_info['total_characters']}")
        print(f"   Custom Mappings: {mapping_info['custom_mappings']}")
        
        print("\n6. Testing Custom Mapping:")
        # Test custom mapping
        print("   Setting custom mapping: anime-waifu -> default_male")
        voice_mapper.set_custom_mapping("anime-waifu", "default_male")
        
        custom_voice = voice_mapper.get_voice_for_character("anime-waifu")
        print(f"   Result: {custom_voice.name} ({custom_voice.gender})")
        
        # Remove custom mapping
        print("   Removing custom mapping...")
        voice_mapper.remove_custom_mapping("anime-waifu")
        
        restored_voice = voice_mapper.get_voice_for_character("anime-waifu")
        print(f"   Restored: {restored_voice.name} ({restored_voice.gender})")
        
        print("\n7. Testing Fallback Logic:")
        # Test with unknown character
        unknown_voice = voice_mapper.get_voice_for_character("unknown_character")
        print(f"   Unknown character fallback: {unknown_voice.name} ({unknown_voice.gender})")
        
        # Test voice model validation
        print("\n8. Voice Model Validation:")
        for voice in voices[:3]:  # Test first 3 voices
            is_valid = voice_manager.validate_voice_model(voice)
            print(f"   {voice.name}: {'✓ Valid' if is_valid else '✗ Invalid'}")
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())