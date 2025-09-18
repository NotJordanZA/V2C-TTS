#!/usr/bin/env python3
"""
Demonstration of the configuration management system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import ConfigManager, AppConfig, AudioConfig, CharacterConfig


def main():
    """Demonstrate configuration management functionality."""
    print("=== Configuration Management System Demo ===\n")
    
    # Initialize config manager
    manager = ConfigManager()
    
    try:
        # Load default configuration
        print("1. Loading default configuration...")
        config = manager.load_config()
        print(f"   ✓ Loaded config with audio sample rate: {config.audio.sample_rate}")
        print(f"   ✓ STT model size: {config.stt.model_size}")
        print(f"   ✓ Character intensity: {config.character.intensity}")
        
        # Demonstrate validation
        print("\n2. Testing configuration validation...")
        print("   ✓ Current config is valid:", manager.validate_config(config))
        
        # Modify configuration
        print("\n3. Modifying configuration...")
        config.audio.sample_rate = 22050
        config.character.intensity = 1.5
        config.stt.model_size = "small"
        print(f"   ✓ Changed sample rate to: {config.audio.sample_rate}")
        print(f"   ✓ Changed character intensity to: {config.character.intensity}")
        print(f"   ✓ Changed STT model to: {config.stt.model_size}")
        
        # Save configuration
        print("\n4. Saving configuration...")
        manager.save_config(config, format="yaml")
        print("   ✓ Configuration saved to user_config.yaml")
        
        # Save as JSON too
        manager.save_config(config, format="json")
        print("   ✓ Configuration also saved as JSON")
        
        # Convert to pipeline config
        print("\n5. Converting to pipeline configuration...")
        pipeline_config = config.to_pipeline_config()
        print(f"   ✓ Pipeline config created with sample rate: {pipeline_config.sample_rate}")
        print(f"   ✓ Pipeline max latency: {pipeline_config.max_latency_ms}ms")
        
        # Demonstrate error handling
        print("\n6. Testing validation errors...")
        try:
            invalid_audio = AudioConfig(sample_rate=-1)
        except Exception as e:
            print(f"   ✓ Caught validation error: {e}")
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())