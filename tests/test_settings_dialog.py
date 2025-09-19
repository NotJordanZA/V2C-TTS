"""
Unit tests for the settings dialog.
"""

import unittest
from unittest.mock import Mock, patch
import tkinter as tk
from pathlib import Path
import tempfile

from src.ui.settings_dialog import SettingsDialog
from src.core.config import ConfigManager, AppConfig, AudioConfig, CharacterConfig, STTConfig, TTSConfig, PerformanceConfig, LoggingConfig
from src.core.interfaces import AudioDevice


class TestSettingsDialog(unittest.TestCase):
    """Test cases for SettingsDialog class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Create test config
        self.config_manager = ConfigManager(self.config_path)
        self.test_config = AppConfig(
            audio=AudioConfig(input_device_id=1, output_device_id=2, sample_rate=44100),
            stt=STTConfig(model_size="base", device="cuda"),
            character=CharacterConfig(intensity=1.5, max_tokens=512, temperature=0.8),
            tts=TTSConfig(device="cuda", speed=1.2),
            performance=PerformanceConfig(max_latency_ms=1500, gpu_memory_fraction=0.7),
            logging=LoggingConfig(level="DEBUG")
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
    
    @patch('src.ui.settings_dialog.get_audio_devices')
    def test_settings_dialog_initialization(self, mock_get_devices):
        """Test settings dialog initialization."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        
        try:
            # Create parent window
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            
            # Create settings dialog
            dialog = SettingsDialog(root, self.config_manager, self.test_config)
            
            # Verify dialog was created
            self.assertIsNotNone(dialog.dialog)
            self.assertEqual(dialog.dialog.title(), "Settings")
            
            # Verify variables were set up
            self.assertIsNotNone(dialog.sample_rate_var)
            self.assertIsNotNone(dialog.intensity_var)
            self.assertIsNotNone(dialog.stt_model_size_var)
            
            # Clean up
            dialog.dialog.destroy()
            root.destroy()
            
        except tk.TclError:
            self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.settings_dialog.get_audio_devices')
    def test_load_current_settings(self, mock_get_devices):
        """Test loading current settings into dialog."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        
        try:
            # Create parent window
            root = tk.Tk()
            root.withdraw()
            
            # Create settings dialog
            dialog = SettingsDialog(root, self.config_manager, self.test_config)
            
            # Verify settings were loaded correctly
            self.assertEqual(dialog.sample_rate_var.get(), 44100)
            self.assertEqual(dialog.intensity_var.get(), 1.5)
            self.assertEqual(dialog.stt_model_size_var.get(), "base")
            self.assertEqual(dialog.stt_device_var.get(), "cuda")
            self.assertEqual(dialog.max_tokens_var.get(), 512)
            self.assertEqual(dialog.temperature_var.get(), 0.8)
            self.assertEqual(dialog.tts_device_var.get(), "cuda")
            self.assertEqual(dialog.tts_speed_var.get(), 1.2)
            self.assertEqual(dialog.max_latency_var.get(), 1500)
            self.assertEqual(dialog.gpu_memory_var.get(), 0.7)
            self.assertEqual(dialog.log_level_var.get(), "DEBUG")
            
            # Clean up
            dialog.dialog.destroy()
            root.destroy()
            
        except tk.TclError:
            self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.settings_dialog.get_audio_devices')
    def test_create_config_from_settings(self, mock_get_devices):
        """Test creating configuration from dialog settings."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        
        try:
            # Create parent window
            root = tk.Tk()
            root.withdraw()
            
            # Create settings dialog
            dialog = SettingsDialog(root, self.config_manager, self.test_config)
            
            # Modify some settings
            dialog.sample_rate_var.set(48000)
            dialog.intensity_var.set(2.0)
            dialog.stt_model_size_var.set("large")
            dialog.max_tokens_var.set(1024)
            
            # Create config from settings
            new_config = dialog._create_config_from_settings()
            
            # Verify new configuration
            self.assertEqual(new_config.audio.sample_rate, 48000)
            self.assertEqual(new_config.character.intensity, 2.0)
            self.assertEqual(new_config.stt.model_size, "large")
            self.assertEqual(new_config.character.max_tokens, 1024)
            
            # Clean up
            dialog.dialog.destroy()
            root.destroy()
            
        except tk.TclError:
            self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.settings_dialog.get_audio_devices')
    def test_device_id_extraction(self, mock_get_devices):
        """Test device ID extraction from device text."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        
        try:
            # Create parent window
            root = tk.Tk()
            root.withdraw()
            
            # Create settings dialog
            dialog = SettingsDialog(root, self.config_manager, self.test_config)
            
            # Test valid device text
            device_id = dialog._extract_device_id("Test Device (ID: 5)")
            self.assertEqual(device_id, 5)
            
            # Test invalid device text
            device_id = dialog._extract_device_id("Invalid Device Text")
            self.assertIsNone(device_id)
            
            # Test empty text
            device_id = dialog._extract_device_id("")
            self.assertIsNone(device_id)
            
            # Clean up
            dialog.dialog.destroy()
            root.destroy()
            
        except tk.TclError:
            self.skipTest("Tkinter not available in test environment")
    
    @patch('src.ui.settings_dialog.get_audio_devices')
    def test_validation(self, mock_get_devices):
        """Test settings validation."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        
        try:
            # Create parent window
            root = tk.Tk()
            root.withdraw()
            
            # Create settings dialog
            dialog = SettingsDialog(root, self.config_manager, self.test_config)
            
            # Test valid settings
            self.assertTrue(dialog._validate_settings())
            
            # Test invalid settings (negative sample rate)
            dialog.sample_rate_var.set(-1000)
            self.assertFalse(dialog._validate_settings())
            
            # Reset to valid
            dialog.sample_rate_var.set(44100)
            self.assertTrue(dialog._validate_settings())
            
            # Test invalid intensity (out of range)
            dialog.intensity_var.set(5.0)
            self.assertFalse(dialog._validate_settings())
            
            # Clean up
            dialog.dialog.destroy()
            root.destroy()
            
        except tk.TclError:
            self.skipTest("Tkinter not available in test environment")
    
    def test_settings_dialog_methods_exist(self):
        """Test that all required settings dialog methods exist."""
        from src.ui.settings_dialog import SettingsDialog
        
        # Check that all required methods exist
        required_methods = [
            '_setup_variables',
            '_setup_ui',
            '_load_current_settings',
            '_load_audio_devices',
            '_create_config_from_settings',
            '_validate_settings',
            '_on_ok',
            '_on_cancel',
            '_on_apply',
            '_on_reset_defaults',
            'show_modal'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(SettingsDialog, method_name), f"Method {method_name} not found")
            self.assertTrue(callable(getattr(SettingsDialog, method_name)), f"Method {method_name} not callable")
    
    @patch('src.ui.settings_dialog.get_audio_devices')
    def test_label_updates(self, mock_get_devices):
        """Test that scale labels update correctly."""
        # Mock audio device functions
        mock_get_devices.return_value = (self.mock_input_devices, self.mock_output_devices)
        
        try:
            # Create parent window
            root = tk.Tk()
            root.withdraw()
            
            # Create settings dialog
            dialog = SettingsDialog(root, self.config_manager, self.test_config)
            
            # Test label update methods
            dialog._update_vad_label(0.75)
            self.assertEqual(dialog.vad_label.cget("text"), "0.75")
            
            dialog._update_intensity_label(1.25)
            self.assertEqual(dialog.intensity_label.cget("text"), "1.25")
            
            dialog._update_temperature_label(0.9)
            self.assertEqual(dialog.temperature_label.cget("text"), "0.90")
            
            dialog._update_speed_label(1.5)
            self.assertEqual(dialog.speed_label.cget("text"), "1.50")
            
            dialog._update_gpu_label(0.85)
            self.assertEqual(dialog.gpu_label.cget("text"), "0.85")
            
            # Clean up
            dialog.dialog.destroy()
            root.destroy()
            
        except tk.TclError:
            self.skipTest("Tkinter not available in test environment")


if __name__ == '__main__':
    unittest.main()