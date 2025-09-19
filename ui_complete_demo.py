#!/usr/bin/env python3
"""
Complete UI demonstration script.

This script demonstrates all the implemented UI functionality including:
- Main application window with character and device selection
- Real-time status display with audio levels and pipeline indicators
- Settings dialog with comprehensive configuration options
- Text processing displays and error messaging

Run this script from the project root directory.
"""

import sys
import os
import logging
import time
import threading
from pathlib import Path

# Set up proper import paths
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
os.environ['PYTHONPATH'] = str(src_path)

def setup_logging():
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def simulate_pipeline_activity(window):
    """Simulate pipeline activity for demonstration."""
    import random
    
    # Simulate audio levels
    for i in range(50):
        if hasattr(window, 'root') and window.root.winfo_exists():
            level = random.uniform(0, 100)
            window.update_audio_level(level)
            time.sleep(0.1)
        else:
            break
    
    # Simulate pipeline stages
    stages = ['audio_capture', 'speech_to_text', 'character_transform', 'text_to_speech', 'audio_output']
    
    for stage in stages:
        if hasattr(window, 'root') and window.root.winfo_exists():
            window.update_pipeline_stage(stage, 'processing')
            time.sleep(1)
            window.update_pipeline_stage(stage, 'complete')
        else:
            break
    
    # Simulate text processing
    if hasattr(window, 'root') and window.root.winfo_exists():
        original_texts = [
            "Hello, how are you today?",
            "This is a test of the voice transformation system.",
            "The weather is really nice outside.",
            "I hope this demonstration is working well."
        ]
        
        transformed_texts = [
            "Konnichiwa~ How are you today desu?",
            "This is a kawaii test of the voice transformation system nya~",
            "The weather is really really nice outside desu!",
            "I hope this demonstration is working well~ Sugoi!"
        ]
        
        for orig, trans in zip(original_texts, transformed_texts):
            if hasattr(window, 'root') and window.root.winfo_exists():
                window.append_original_text(orig)
                window.append_transformed_text(trans)
                time.sleep(2)
            else:
                break

def main():
    """Main demo function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        print("🎬 Starting Complete UI Demo")
        print("=" * 50)
        
        # Import UI components
        from src.ui.main_window import MainWindow
        from src.ui.settings_dialog import SettingsDialog
        from src.core.config import ConfigManager
        
        print("✅ UI components imported successfully")
        
        # Initialize configuration manager
        config_manager = ConfigManager()
        print("✅ Configuration manager initialized")
        
        # Create main window
        print("🖥️  Creating main window...")
        window = MainWindow(config_manager)
        print("✅ Main window created successfully")
        
        # Set up callbacks to demonstrate functionality
        def on_character_changed(character_name):
            logger.info(f"Character changed to: {character_name}")
            window.show_info_message(f"Character changed to: {character_name}")
        
        def on_input_device_changed(device_id):
            logger.info(f"Input device changed to ID: {device_id}")
            window.show_info_message(f"Input device changed to ID: {device_id}")
        
        def on_output_device_changed(device_id):
            logger.info(f"Output device changed to ID: {device_id}")
            window.show_info_message(f"Output device changed to ID: {device_id}")
        
        def on_start_pipeline():
            logger.info("Pipeline start requested")
            window.show_info_message("Pipeline started successfully!")
            window.clear_text_displays()
            window.reset_pipeline_status()
            
            # Start simulation in background thread
            simulation_thread = threading.Thread(target=simulate_pipeline_activity, args=(window,))
            simulation_thread.daemon = True
            simulation_thread.start()
        
        def on_stop_pipeline():
            logger.info("Pipeline stop requested")
            window.show_warning_message("Pipeline stopped by user")
            window.reset_pipeline_status()
        
        # Connect callbacks
        window.on_character_changed = on_character_changed
        window.on_input_device_changed = on_input_device_changed
        window.on_output_device_changed = on_output_device_changed
        window.on_start_pipeline = on_start_pipeline
        window.on_stop_pipeline = on_stop_pipeline
        
        print("✅ Callbacks connected")
        
        # Show initial demo messages
        window.show_info_message("Welcome to the Voice Character Transformation UI Demo!")
        window.show_info_message("Try selecting different characters and audio devices")
        window.show_info_message("Click 'Start Pipeline' to see the status indicators in action")
        window.show_info_message("Open 'Settings' to configure advanced options")
        
        # Add some demo text
        window.update_original_text("Welcome to the demo! This shows original speech-to-text output.")
        window.update_transformed_text("Welcome to the demo desu~ This shows kawaii transformed text nya!")
        
        print("✅ Demo setup complete")
        print("\n🎮 Demo Instructions:")
        print("1. Try selecting different characters from the dropdown")
        print("2. Change audio input/output devices")
        print("3. Click 'Start Pipeline' to see status indicators")
        print("4. Open 'Settings' to explore configuration options")
        print("5. Watch the real-time status displays update")
        print("\n🚀 Starting UI demo...")
        
        # Run the application
        window.run()
        
        print("✅ Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())