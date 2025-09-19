"""
Unit tests for the main application window.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from pathlib import Path
import tempfile
import json

from src.ui.main_window import MainWindow
from src.core.config import ConfigManager, AppConfig, AudioConfig, CharacterConfig, STTConfig, TTSConfig, PerformanceConfig, LoggingConfig
from src.core.interfaces import AudioDevice


class TestMainWindow(unittest.TestCase):
    """Test cases for MainWindow class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        self.characters_dir = Path(self.temp_dir) / "characters"
        self.characters_dir.mkdir(exist_ok=True)
        
        # Create test character profiles
        self._create_test_characters()
        
        # Create test config
        self.config_manager = ConfigManager(self.config_path)
        self.test_config = AppConfig(
            audio=AudioConfig(input_device_id=1, output_device_id=2),
            stt=STTConfig(),
            character=CharacterConfig(default_character="test-character"),
            tts=TTSConfig(),
            performance=PerformanceConfig(),
            logging=LoggingConfig()
        )
        
        # Mock audio devices
        self.mock_input_devices = [
            AudioDevice(id=0, name="Test Microphone 1", channels=1, sample_rate=16000, is_input=True),
            AudioDevice(id=1, name="Test Microphone 2", channels=2, sample_rate=44100, is_input=True)
        ]
        
        self.mock_output_devices = [
            AudioDevice(id=2, name="Test Speaker 1", channels=2, sample_rate=44100, is_input=False),
            AudioDevice(id=3, name="Test Speaker 2", channels=2, sample_rate=48000, is_input=False)
        ]
    
    def _create_test_characters(self):
        """Create test character profile files."""
        test_character = {
            "name": "test-character",
            "description": "A test character for unit testing",
            "personality_traits": ["test", "friendly"],
            "speech_patterns": {"hello": "hi there"},
            "vocabulary_preferences": {"greetings": ["hello", "hi"]},
            "transformation_prompt": "Transform this text: {text}",
            "voice_model_path": "test/voice.pth",
            "intensity_multiplier": 1.0
        }
        
        with open(self.characters_dir / "test-character.json", 'w') as f:
            json.dump(test_character, f)
        
        default_character = {
            "name": "default",
            "description": "Default test character",
            "personality_traits": ["neutral"],
            "speech_patterns": {},
            "vocabulary_preferences": {},
            "transformation_prompt": "Repeat: {text}",
            "voice_model_path": "",
            "intensity_multiplier": 1.0
        }
        
        with open(self.characters_dir / "default.json", 'w') as f:
            json.dump(default_character, f)
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_main_window_initialization(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test main window initialization."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        # Mock character manager
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["test-character", "default"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        # Mock config manager
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            try:
                # Create main window (without running mainloop)
                window = MainWindow(self.config_manager)
                
                # Verify initialization
                self.assertIsNotNone(window.root)
                self.assertEqual(window.root.title(), "Voice Character Transformation")
                self.assertIsNotNone(window.character_combo)
                self.assertIsNotNone(window.input_device_combo)
                self.assertIsNotNone(window.output_device_combo)
                
                # Clean up
                window.root.destroy()
            except tk.TclError:
                # Skip test if Tkinter is not available in test environment
                self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_character_selection(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test character selection functionality."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        # Mock character manager
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["test-character", "default"]
        
        # Mock character profile
        mock_profile = Mock()
        mock_profile.description = "Test character description"
        mock_char_manager_instance.load_profile.return_value = mock_profile
        mock_char_manager.return_value = mock_char_manager_instance
        
        # Mock config manager
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            window = MainWindow(self.config_manager)
            
            # Test character selection
            callback_called = False
            selected_character = None
            
            def character_callback(character_name):
                nonlocal callback_called, selected_character
                callback_called = True
                selected_character = character_name
            
            window.on_character_changed = character_callback
            
            # Simulate character selection
            window.selected_character.set("test-character")
            window.root.update()  # Process events
            
            # Verify callback was called
            self.assertTrue(callback_called)
            self.assertEqual(selected_character, "test-character")
            
            # Clean up
            window.root.destroy()
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_audio_device_selection(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test audio device selection functionality."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        # Mock character manager
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["test-character"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        # Mock config manager
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            window = MainWindow(self.config_manager)
            
            # Test input device selection
            input_callback_called = False
            selected_input_id = None
            
            def input_callback(device_id):
                nonlocal input_callback_called, selected_input_id
                input_callback_called = True
                selected_input_id = device_id
            
            window.on_input_device_changed = input_callback
            
            # Simulate input device selection
            window.selected_input_device.set("Test Microphone 1 (ID: 0)")
            window.root.update()  # Process events
            
            # Verify callback was called
            self.assertTrue(input_callback_called)
            self.assertEqual(selected_input_id, 0)
            
            # Test output device selection
            output_callback_called = False
            selected_output_id = None
            
            def output_callback(device_id):
                nonlocal output_callback_called, selected_output_id
                output_callback_called = True
                selected_output_id = device_id
            
            window.on_output_device_changed = output_callback
            
            # Simulate output device selection
            window.selected_output_device.set("Test Speaker 2 (ID: 3)")
            window.root.update()  # Process events
            
            # Verify callback was called
            self.assertTrue(output_callback_called)
            self.assertEqual(selected_output_id, 3)
            
            # Clean up
            window.root.destroy()
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_pipeline_control_buttons(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test pipeline control button functionality."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        # Mock character manager
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["test-character"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        # Mock config manager
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            try:
                window = MainWindow(self.config_manager)
                
                # Set up selections for valid start
                window.selected_character.set("test-character")
                window.selected_input_device.set("Test Microphone 1 (ID: 0)")
                window.selected_output_device.set("Test Speaker 1 (ID: 2)")
                
                # Test start pipeline
                start_callback_called = False
                
                def start_callback():
                    nonlocal start_callback_called
                    start_callback_called = True
                
                window.on_start_pipeline = start_callback
                
                # Simulate start button click
                window._on_start_clicked()
                
                # Verify callback was called and button states changed
                self.assertTrue(start_callback_called)
                self.assertEqual(str(window.start_button['state']), 'disabled')
                self.assertEqual(str(window.stop_button['state']), 'normal')
                
                # Test stop pipeline
                stop_callback_called = False
                
                def stop_callback():
                    nonlocal stop_callback_called
                    stop_callback_called = True
                
                window.on_stop_pipeline = stop_callback
                
                # Simulate stop button click
                window._on_stop_clicked()
                
                # Verify callback was called and button states changed
                self.assertTrue(stop_callback_called)
                self.assertEqual(str(window.start_button['state']), 'normal')
                self.assertEqual(str(window.stop_button['state']), 'disabled')
                
                # Clean up
                window.root.destroy()
            except tk.TclError:
                # Skip test if Tkinter is not available in test environment
                self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_device_id_extraction(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test device ID extraction from device text."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        # Mock character manager
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["test-character"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        # Mock config manager
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            window = MainWindow(self.config_manager)
            
            # Test valid device text
            device_id = window._extract_device_id("Test Device (ID: 5)")
            self.assertEqual(device_id, 5)
            
            # Test invalid device text
            device_id = window._extract_device_id("Invalid Device Text")
            self.assertIsNone(device_id)
            
            # Test empty text
            device_id = window._extract_device_id("")
            self.assertIsNone(device_id)
            
            # Clean up
            window.root.destroy()
    
    def test_device_id_extraction_standalone(self):
        """Test device ID extraction method without UI initialization."""
        from src.ui.main_window import MainWindow
        
        # Test the static method directly without creating a window
        # Create a mock window instance just for the method
        mock_window = Mock(spec=MainWindow)
        mock_window._extract_device_id = MainWindow._extract_device_id.__get__(mock_window)
        
        # Test valid device text
        device_id = mock_window._extract_device_id("Test Device (ID: 5)")
        self.assertEqual(device_id, 5)
        
        # Test invalid device text
        device_id = mock_window._extract_device_id("Invalid Device Text")
        self.assertIsNone(device_id)
        
        # Test empty text
        device_id = mock_window._extract_device_id("")
        self.assertIsNone(device_id)


if __name__ == '__main__':
    unittest.main()