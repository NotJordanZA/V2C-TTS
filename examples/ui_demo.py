#!/usr/bin/env python3
"""
Demo script for the main application window.

This script demonstrates the main UI window functionality including
character selection, audio device configuration, and pipeline control.

Run this script from the project root directory using:
python -m examples.ui_demo

Or run directly with proper PYTHONPATH:
PYTHONPATH=src python examples/ui_demo.py
"""

import sys
import logging
import os
from pathlib import Path

def setup_imports():
    """Set up proper import paths."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    
    # Add src to Python path if not already there
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set PYTHONPATH environment variable for subprocess imports
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(src_path) not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{src_path}{os.pathsep}{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = str(src_path)

# Set up imports before importing our modules
setup_imports()

# Now import our modules
from src.ui.main_window import MainWindow
from src.core.config import ConfigManager


def setup_logging():
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main demo function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Create main window
        logger.info("Creating main window...")
        window = MainWindow(config_manager)
        
        # Set up callbacks to demonstrate functionality
        def on_character_changed(character_name):
            logger.info(f"Character changed to: {character_name}")
        
        def on_input_device_changed(device_id):
            logger.info(f"Input device changed to ID: {device_id}")
        
        def on_output_device_changed(device_id):
            logger.info(f"Output device changed to ID: {device_id}")
        
        def on_start_pipeline():
            logger.info("Pipeline start requested")
            logger.info(f"Selected character: {window.get_selected_character()}")
            logger.info(f"Selected input device ID: {window.get_selected_input_device_id()}")
            logger.info(f"Selected output device ID: {window.get_selected_output_device_id()}")
        
        def on_stop_pipeline():
            logger.info("Pipeline stop requested")
        
        # Connect callbacks
        window.on_character_changed = on_character_changed
        window.on_input_device_changed = on_input_device_changed
        window.on_output_device_changed = on_output_device_changed
        window.on_start_pipeline = on_start_pipeline
        window.on_stop_pipeline = on_stop_pipeline
        
        # Run the application
        logger.info("Starting main window...")
        window.run()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()