#!/usr/bin/env python3
"""
Simple test script for the main application window.

This script tests the UI window functionality by running it as a module.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set PYTHONPATH for subprocess imports
os.environ['PYTHONPATH'] = str(src_path)

def main():
    """Test the main window functionality."""
    try:
        print("Testing UI imports...")
        
        # Test imports
        from src.ui.main_window import MainWindow
        from src.core.config import ConfigManager
        print("✅ Imports successful")
        
        # Test configuration manager
        config_manager = ConfigManager()
        print("✅ ConfigManager created")
        
        # Test main window creation (without showing)
        print("Creating main window...")
        window = MainWindow(config_manager)
        print("✅ MainWindow created successfully")
        
        # Test getter methods
        print("Testing getter methods...")
        character = window.get_selected_character()
        input_device = window.get_selected_input_device_id()
        output_device = window.get_selected_output_device_id()
        print(f"✅ Selected character: {character}")
        print(f"✅ Selected input device ID: {input_device}")
        print(f"✅ Selected output device ID: {output_device}")
        
        # Test status display methods
        print("Testing status display methods...")
        
        # Test audio level update
        window.update_audio_level(75.5)
        print("✅ Audio level updated")
        
        # Test pipeline stage updates
        window.update_pipeline_stage('audio_capture', 'processing')
        window.update_pipeline_stage('speech_to_text', 'complete')
        window.update_pipeline_stage('character_transform', 'error', 'Connection failed')
        print("✅ Pipeline stages updated")
        
        # Test text displays
        window.update_original_text("Hello, this is a test message!")
        window.update_transformed_text("Konnichiwa~ This is a kawaii test message desu!")
        print("✅ Text displays updated")
        
        # Test message displays
        window.show_info_message("Pipeline started successfully")
        window.show_warning_message("Audio level is high")
        window.show_error_message("Failed to connect to TTS service")
        print("✅ Messages displayed")
        
        # Test reset functions
        window.reset_pipeline_status()
        window.clear_text_displays()
        print("✅ Reset functions tested")
        
        # Test settings dialog creation (without showing)
        print("Testing settings dialog...")
        from src.ui.settings_dialog import SettingsDialog
        
        # Create settings dialog (but don't show it)
        settings_dialog = SettingsDialog(window.root, config_manager, window.config)
        
        # Test some settings dialog methods
        settings_dialog._update_intensity_label(1.5)
        settings_dialog._update_vad_label(0.75)
        print("✅ Settings dialog created and tested")
        
        # Test validation
        is_valid = settings_dialog._validate_settings()
        print(f"✅ Settings validation: {is_valid}")
        
        # Clean up settings dialog
        settings_dialog.dialog.destroy()
        print("✅ Settings dialog cleaned up")
        
        # Clean up main window
        window.root.destroy()
        print("✅ Main window cleaned up successfully")
        
        print("\n🎉 All UI tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())