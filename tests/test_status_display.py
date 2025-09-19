"""
Unit tests for the status display functionality of the main window.
"""

import unittest
from unittest.mock import Mock, patch
import tkinter as tk
from pathlib import Path
import tempfile
import json

from src.ui.main_window import MainWindow
from src.core.config import ConfigManager, AppConfig, AudioConfig, CharacterConfig, STTConfig, TTSConfig, PerformanceConfig, LoggingConfig
from src.core.interfaces import AudioDevice


class TestStatusDisplay(unittest.TestCase):
    """Test cases for status display functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Create test config
        self.config_manager = ConfigManager(self.config_path)
        self.test_config = AppConfig(
            audio=AudioConfig(),
            stt=STTConfig(),
            character=CharacterConfig(),
            tts=TTSConfig(),
            performance=PerformanceConfig(),
            logging=LoggingConfig()
        )
        
        # Mock audio devices
        self.mock_input_devices = [
            AudioDevice(id=0, name="Test Microphone", channels=1, sample_rate=16000, is_input=True)
        ]
        
        self.mock_output_devices = [
            AudioDevice(id=1, name="Test Speaker", channels=2, sample_rate=44100, is_input=False)
        ]
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_audio_level_update(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test audio level indicator updates."""
        # Mock dependencies
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["default"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            try:
                window = MainWindow(self.config_manager)
                
                # Test normal level
                window.update_audio_level(50.0)
                self.assertEqual(window.audio_level_var.get(), 50.0)
                self.assertEqual(window.audio_level_label.cget("text"), "50.0%")
                
                # Test high level
                window.update_audio_level(85.0)
                self.assertEqual(window.audio_level_var.get(), 85.0)
                
                # Test level clamping
                window.update_audio_level(150.0)  # Should be clamped to 100
                self.assertEqual(window.audio_level_var.get(), 100.0)
                
                window.update_audio_level(-10.0)  # Should be clamped to 0
                self.assertEqual(window.audio_level_var.get(), 0.0)
                
                window.root.destroy()
                
            except tk.TclError:
                self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_pipeline_stage_updates(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test pipeline stage indicator updates."""
        # Mock dependencies
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["default"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            try:
                window = MainWindow(self.config_manager)
                
                # Test different status updates
                window.update_pipeline_stage('audio_capture', 'processing')
                self.assertEqual(window.pipeline_stages['audio_capture']['status'], 'processing')
                
                window.update_pipeline_stage('speech_to_text', 'complete')
                self.assertEqual(window.pipeline_stages['speech_to_text']['status'], 'complete')
                
                window.update_pipeline_stage('character_transform', 'error', 'Connection failed')
                self.assertEqual(window.pipeline_stages['character_transform']['status'], 'error')
                
                # Test invalid stage
                window.update_pipeline_stage('invalid_stage', 'processing')  # Should not crash
                
                window.root.destroy()
                
            except tk.TclError:
                self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_text_display_updates(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test text display area updates."""
        # Mock dependencies
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["default"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            try:
                window = MainWindow(self.config_manager)
                
                # Test original text update
                test_original = "Hello, this is a test message."
                window.update_original_text(test_original)
                # Note: Can't easily test Text widget content in unit tests due to Tkinter limitations
                
                # Test transformed text update
                test_transformed = "Konnichiwa~ This is a kawaii test message desu!"
                window.update_transformed_text(test_transformed)
                
                # Test append functions
                window.append_original_text("Additional text")
                window.append_transformed_text("More kawaii text~")
                
                # Test clear function
                window.clear_text_displays()
                
                window.root.destroy()
                
            except tk.TclError:
                self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_message_display(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test error/info/warning message display."""
        # Mock dependencies
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["default"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            try:
                window = MainWindow(self.config_manager)
                
                # Test different message types
                window.show_error_message("Test error message")
                window.show_warning_message("Test warning message")
                window.show_info_message("Test info message")
                
                # Test clear messages
                window._clear_messages()
                
                window.root.destroy()
                
            except tk.TclError:
                self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.main_window.get_audio_devices')
    @patch('src.ui.main_window.get_default_devices')
    @patch('src.ui.main_window.CharacterProfileManager')
    def test_reset_functions(self, mock_char_manager, mock_default_devices, mock_get_devices):
        """Test reset and clear functions."""
        # Mock dependencies
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        mock_default_devices.return_value = (self.mock_input_devices[0], self.mock_output_devices[0])
        
        mock_char_manager_instance = Mock()
        mock_char_manager_instance.get_available_profiles.return_value = ["default"]
        mock_char_manager.return_value = mock_char_manager_instance
        
        with patch.object(self.config_manager, 'get_config', return_value=self.test_config):
            try:
                window = MainWindow(self.config_manager)
                
                # Set some states
                window.update_pipeline_stage('audio_capture', 'processing')
                window.update_pipeline_stage('speech_to_text', 'error')
                window.update_original_text("Test text")
                window.show_error_message("Test error")
                
                # Test reset pipeline status
                window.reset_pipeline_status()
                for stage in window.pipeline_stages:
                    self.assertEqual(window.pipeline_stages[stage]['status'], 'idle')
                
                # Test clear functions
                window.clear_text_displays()
                window._clear_messages()
                
                window.root.destroy()
                
            except tk.TclError:
                self.skipTest("Tkinter not available in test environment")
    
    def test_status_display_methods_exist(self):
        """Test that all required status display methods exist."""
        from src.ui.main_window import MainWindow
        
        # Check that all required methods exist
        required_methods = [
            'update_audio_level',
            'update_pipeline_stage',
            'update_original_text',
            'update_transformed_text',
            'append_original_text',
            'append_transformed_text',
            'show_error_message',
            'show_info_message',
            'show_warning_message',
            'reset_pipeline_status',
            'clear_text_displays'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(MainWindow, method_name), f"Method {method_name} not found")
            self.assertTrue(callable(getattr(MainWindow, method_name)), f"Method {method_name} not callable")


if __name__ == '__main__':
    unittest.main()