#!/usr/bin/env python3
"""
Simple demo script that shows the core functionality working.

This script demonstrates the key components without the complex
application lifecycle management that has import issues.
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def setup_logging():
    """Set up basic logging."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def demo_audio_system():
    """Demo the audio system."""
    print("\n🎵 Testing Audio System")
    print("-" * 30)
    
    try:
        from audio.device_manager import AudioDeviceManager
        
        device_manager = AudioDeviceManager()
        input_devices = device_manager.get_input_devices()
        output_devices = device_manager.get_output_devices()
        
        print(f"✅ Found {len(input_devices)} input devices")
        print(f"✅ Found {len(output_devices)} output devices")
        
        # Show first few devices
        if input_devices:
            print("📥 Input devices:")
            for i, device in enumerate(input_devices[:3]):
                print(f"   {i}: {device['name']}")
        
        if output_devices:
            print("📤 Output devices:")
            for i, device in enumerate(output_devices[:3]):
                print(f"   {i}: {device['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio system error: {e}")
        return False

async def demo_character_system():
    """Demo the character system."""
    print("\n🎭 Testing Character System")
    print("-" * 30)
    
    try:
        from character.profile import CharacterProfileManager
        
        profile_manager = CharacterProfileManager("characters")
        profiles = profile_manager.list_profiles()
        
        print(f"✅ Found {len(profiles)} character profiles:")
        for profile_name in profiles:
            print(f"   - {profile_name}")
        
        # Load and show a character
        if profiles:
            character = profile_manager.load_profile(profiles[0])
            print(f"\n📋 Character: {character.name}")
            print(f"   Description: {character.description}")
            print(f"   Traits: {', '.join(character.personality_traits[:3])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Character system error: {e}")
        return False

async def demo_config_system():
    """Demo the configuration system."""
    print("\n⚙️  Testing Configuration System")
    print("-" * 30)
    
    try:
        from core.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print("✅ Configuration loaded successfully")
        print(f"   Audio sample rate: {config.audio.sample_rate}")
        print(f"   Audio chunk size: {config.audio.chunk_size}")
        print(f"   STT model size: {config.stt.model_size}")
        print(f"   Max latency: {config.performance.max_latency_ms}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system error: {e}")
        return False

async def demo_profiler_system():
    """Demo the profiler system."""
    print("\n📊 Testing Profiler System")
    print("-" * 30)
    
    try:
        from core.profiler import SystemProfiler
        
        profiler = SystemProfiler(sampling_interval=0.5)
        profiler.start_profiling()
        
        print("✅ Profiler started")
        
        # Let it collect some data
        await asyncio.sleep(2)
        
        # Get system summary
        summary = profiler.get_system_summary()
        if summary:
            print(f"   CPU usage: {summary['cpu']['current_percent']:.1f}%")
            print(f"   Memory usage: {summary['memory']['current_mb']:.1f}MB")
        
        profiler.stop_profiling()
        print("✅ Profiler stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Profiler system error: {e}")
        return False

async def demo_quality_manager():
    """Demo the quality management system."""
    print("\n🎯 Testing Quality Manager")
    print("-" * 30)
    
    try:
        from core.profiler import SystemProfiler
        from core.quality_manager import QualityManager, QualityLevel
        from core.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        profiler = SystemProfiler()
        
        quality_manager = QualityManager(profiler, config)
        
        print("✅ Quality manager initialized")
        
        # Show available quality levels
        levels = quality_manager.get_available_quality_levels()
        print(f"   Available quality levels: {len(levels)}")
        for level_name, info in levels.items():
            print(f"   - {level_name}: {info['name']}")
        
        # Test quality adjustment
        quality_manager.force_quality_level(QualityLevel.MEDIUM)
        current_info = quality_manager.get_current_quality_info()
        print(f"   Current level: {current_info['current_level']}")
        print(f"   Max latency: {current_info['profile']['max_latency_ms']}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality manager error: {e}")
        return False

async def main():
    """Run the simple demo."""
    print("🎵 Voice Character Transformation - Simple Demo")
    print("=" * 60)
    
    setup_logging()
    
    # Test each system
    systems = [
        ("Configuration System", demo_config_system),
        ("Audio System", demo_audio_system),
        ("Character System", demo_character_system),
        ("Profiler System", demo_profiler_system),
        ("Quality Manager", demo_quality_manager),
    ]
    
    results = {}
    
    for system_name, demo_func in systems:
        try:
            success = await demo_func()
            results[system_name] = success
        except Exception as e:
            print(f"❌ {system_name} failed: {e}")
            results[system_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEMO SUMMARY")
    print("=" * 60)
    
    total_systems = len(results)
    working_systems = sum(1 for success in results.values() if success)
    
    print(f"Systems tested: {total_systems}")
    print(f"Working systems: {working_systems}")
    print(f"Success rate: {working_systems/total_systems*100:.1f}%")
    
    print("\nDetailed results:")
    for system_name, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"  {status} {system_name}")
    
    if working_systems == total_systems:
        print("\n🎉 All systems are working correctly!")
        print("The voice character transformation system is ready to use.")
    else:
        print(f"\n⚠️  {total_systems - working_systems} system(s) need attention.")
        print("Check the error messages above for details.")
    
    print("\n💡 To see the full UI in action, run:")
    print("   python ui_complete_demo.py")
    print("   or")
    print("   python run_app.py ui")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()