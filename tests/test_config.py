"""
Unit tests for configuration management system.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.core.config import (
    AudioConfig, STTConfig, CharacterConfig, TTSConfig, 
    PerformanceConfig, LoggingConfig, AppConfig, ConfigManager,
    ConfigValidationError
)


class TestAudioConfig:
    """Test AudioConfig validation."""
    
    def test_valid_audio_config(self):
        """Test valid audio configuration."""
        config = AudioConfig(
            sample_rate=16000,
            chunk_size=1024,
            input_device_id=0,
            output_device_id=1,
            vad_threshold=0.5
        )
        assert config.sample_rate == 16000
        assert config.chunk_size == 1024
    
    def test_invalid_sample_rate(self):
        """Test invalid sample rate validation."""
        with pytest.raises(ConfigValidationError, match="sample_rate must be positive"):
            AudioConfig(sample_rate=-1)
        
        with pytest.raises(ConfigValidationError, match="sample_rate .* not supported"):
            AudioConfig(sample_rate=12345)
    
    def test_invalid_chunk_size(self):
        """Test invalid chunk size validation."""
        with pytest.raises(ConfigValidationError, match="chunk_size must be between"):
            AudioConfig(chunk_size=0)
        
        with pytest.raises(ConfigValidationError, match="chunk_size must be between"):
            AudioConfig(chunk_size=10000)
    
    def test_invalid_vad_threshold(self):
        """Test invalid VAD threshold validation."""
        with pytest.raises(ConfigValidationError, match="vad_threshold must be between"):
            AudioConfig(vad_threshold=-0.1)
        
        with pytest.raises(ConfigValidationError, match="vad_threshold must be between"):
            AudioConfig(vad_threshold=1.1)


class TestSTTConfig:
    """Test STTConfig validation."""
    
    def test_valid_stt_config(self):
        """Test valid STT configuration."""
        config = STTConfig(model_size="base", device="cuda", language="en")
        assert config.model_size == "base"
        assert config.device == "cuda"
    
    def test_invalid_model_size(self):
        """Test invalid model size validation."""
        with pytest.raises(ConfigValidationError, match="model_size must be one of"):
            STTConfig(model_size="invalid")
    
    def test_invalid_device(self):
        """Test invalid device validation."""
        with pytest.raises(ConfigValidationError, match="device must be 'cuda' or 'cpu'"):
            STTConfig(device="gpu")


class TestCharacterConfig:
    """Test CharacterConfig validation."""
    
    def test_valid_character_config(self):
        """Test valid character configuration."""
        config = CharacterConfig(
            default_character="anime_waifu",
            intensity=1.0,
            llm_model_path="models/test.gguf",
            max_tokens=256,
            temperature=0.7
        )
        assert config.intensity == 1.0
        assert config.max_tokens == 256
    
    def test_invalid_intensity(self):
        """Test invalid intensity validation."""
        with pytest.raises(ConfigValidationError, match="intensity must be between"):
            CharacterConfig(intensity=-0.1)
        
        with pytest.raises(ConfigValidationError, match="intensity must be between"):
            CharacterConfig(intensity=2.1)
    
    def test_invalid_max_tokens(self):
        """Test invalid max_tokens validation."""
        with pytest.raises(ConfigValidationError, match="max_tokens must be between"):
            CharacterConfig(max_tokens=0)
        
        with pytest.raises(ConfigValidationError, match="max_tokens must be between"):
            CharacterConfig(max_tokens=5000)
    
    def test_invalid_temperature(self):
        """Test invalid temperature validation."""
        with pytest.raises(ConfigValidationError, match="temperature must be between"):
            CharacterConfig(temperature=-0.1)
        
        with pytest.raises(ConfigValidationError, match="temperature must be between"):
            CharacterConfig(temperature=2.1)
    
    def test_empty_model_path(self):
        """Test empty model path validation."""
        with pytest.raises(ConfigValidationError, match="llm_model_path cannot be empty"):
            CharacterConfig(llm_model_path="")


class TestTTSConfig:
    """Test TTSConfig validation."""
    
    def test_valid_tts_config(self):
        """Test valid TTS configuration."""
        config = TTSConfig(
            model_path="models/tts.pth",
            device="cuda",
            sample_rate=22050,
            speed=1.0
        )
        assert config.sample_rate == 22050
        assert config.speed == 1.0
    
    def test_invalid_device(self):
        """Test invalid device validation."""
        with pytest.raises(ConfigValidationError, match="device must be 'cuda' or 'cpu'"):
            TTSConfig(device="gpu")
    
    def test_invalid_sample_rate(self):
        """Test invalid sample rate validation."""
        with pytest.raises(ConfigValidationError, match="sample_rate .* not supported"):
            TTSConfig(sample_rate=12000)
    
    def test_invalid_speed(self):
        """Test invalid speed validation."""
        with pytest.raises(ConfigValidationError, match="speed must be between"):
            TTSConfig(speed=0.05)
        
        with pytest.raises(ConfigValidationError, match="speed must be between"):
            TTSConfig(speed=4.0)
    
    def test_empty_model_path(self):
        """Test empty model path validation."""
        with pytest.raises(ConfigValidationError, match="model_path cannot be empty"):
            TTSConfig(model_path="")


class TestPerformanceConfig:
    """Test PerformanceConfig validation."""
    
    def test_valid_performance_config(self):
        """Test valid performance configuration."""
        config = PerformanceConfig(
            max_latency_ms=2000,
            gpu_memory_fraction=0.8,
            enable_model_offloading=True,
            batch_size=1
        )
        assert config.max_latency_ms == 2000
        assert config.gpu_memory_fraction == 0.8
    
    def test_invalid_max_latency(self):
        """Test invalid max latency validation."""
        with pytest.raises(ConfigValidationError, match="max_latency_ms must be between"):
            PerformanceConfig(max_latency_ms=0)
        
        with pytest.raises(ConfigValidationError, match="max_latency_ms must be between"):
            PerformanceConfig(max_latency_ms=15000)
    
    def test_invalid_gpu_memory_fraction(self):
        """Test invalid GPU memory fraction validation."""
        with pytest.raises(ConfigValidationError, match="gpu_memory_fraction must be between"):
            PerformanceConfig(gpu_memory_fraction=0.05)
        
        with pytest.raises(ConfigValidationError, match="gpu_memory_fraction must be between"):
            PerformanceConfig(gpu_memory_fraction=1.1)
    
    def test_invalid_batch_size(self):
        """Test invalid batch size validation."""
        with pytest.raises(ConfigValidationError, match="batch_size must be between"):
            PerformanceConfig(batch_size=0)
        
        with pytest.raises(ConfigValidationError, match="batch_size must be between"):
            PerformanceConfig(batch_size=50)


class TestLoggingConfig:
    """Test LoggingConfig validation."""
    
    def test_valid_logging_config(self):
        """Test valid logging configuration."""
        config = LoggingConfig(
            level="INFO",
            file="logs/test.log",
            max_file_size="10MB",
            backup_count=5
        )
        assert config.level == "INFO"
        assert config.backup_count == 5
    
    def test_invalid_level(self):
        """Test invalid logging level validation."""
        with pytest.raises(ConfigValidationError, match="level must be one of"):
            LoggingConfig(level="INVALID")
    
    def test_invalid_backup_count(self):
        """Test invalid backup count validation."""
        with pytest.raises(ConfigValidationError, match="backup_count must be between"):
            LoggingConfig(backup_count=-1)
        
        with pytest.raises(ConfigValidationError, match="backup_count must be between"):
            LoggingConfig(backup_count=150)
    
    def test_empty_file_path(self):
        """Test empty file path validation."""
        with pytest.raises(ConfigValidationError, match="file path cannot be empty"):
            LoggingConfig(file="")
    
    def test_invalid_max_file_size(self):
        """Test invalid max file size format validation."""
        with pytest.raises(ConfigValidationError, match="max_file_size must be in format"):
            LoggingConfig(max_file_size="invalid")
        
        # Valid formats should work
        LoggingConfig(max_file_size="10MB")
        LoggingConfig(max_file_size="1GB")
        LoggingConfig(max_file_size="500KB")


class TestAppConfig:
    """Test AppConfig functionality."""
    
    def test_valid_app_config(self):
        """Test valid app configuration creation."""
        config = AppConfig(
            audio=AudioConfig(),
            stt=STTConfig(),
            character=CharacterConfig(),
            tts=TTSConfig(),
            performance=PerformanceConfig(),
            logging=LoggingConfig()
        )
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.stt, STTConfig)
    
    def test_to_pipeline_config(self):
        """Test conversion to PipelineConfig."""
        app_config = AppConfig(
            audio=AudioConfig(input_device_id=1, sample_rate=16000, chunk_size=512),
            stt=STTConfig(model_size="small"),
            character=CharacterConfig(llm_model_path="test.gguf"),
            tts=TTSConfig(model_path="test.pth"),
            performance=PerformanceConfig(max_latency_ms=1500),
            logging=LoggingConfig()
        )
        
        pipeline_config = app_config.to_pipeline_config()
        assert pipeline_config.audio_device_id == 1
        assert pipeline_config.sample_rate == 16000
        assert pipeline_config.chunk_size == 512
        assert pipeline_config.stt_model_size == "small"
        assert pipeline_config.max_latency_ms == 1500


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_init_with_default_path(self):
        """Test ConfigManager initialization with default path."""
        manager = ConfigManager()
        assert manager.config_path == Path("config/default_config.yaml")
        assert manager.user_config_path == Path("config/user_config.yaml")
    
    def test_init_with_custom_path(self):
        """Test ConfigManager initialization with custom path."""
        custom_path = Path("custom/config.yaml")
        manager = ConfigManager(custom_path)
        assert manager.config_path == custom_path
    
    @patch('builtins.open', new_callable=mock_open, read_data="""
audio:
  sample_rate: 16000
  chunk_size: 1024
stt:
  model_size: base
character:
  llm_model_path: test.gguf
tts:
  model_path: test.pth
performance: {}
logging: {}
""")
    @patch('pathlib.Path.exists')
    def test_load_yaml_config(self, mock_exists, mock_file):
        """Test loading YAML configuration."""
        mock_exists.return_value = True
        
        manager = ConfigManager()
        config = manager.load_config()
        
        assert isinstance(config, AppConfig)
        assert config.audio.sample_rate == 16000
        assert config.stt.model_size == "base"
    
    def test_load_json_config(self):
        """Test loading JSON configuration."""
        with patch.object(ConfigManager, '_load_config_file') as mock_load:
            mock_load.side_effect = [
                {"audio": {"sample_rate": 16000}, "stt": {}, "character": {"llm_model_path": "test"}, 
                 "tts": {"model_path": "test"}, "performance": {}, "logging": {}},
                {"audio": {"sample_rate": 22050}}
            ]
            with patch('pathlib.Path.exists', return_value=True):
                manager = ConfigManager(Path("config/test.json"))
                config = manager.load_config()
                assert config.audio.sample_rate == 22050
    
    def test_save_yaml_config(self):
        """Test saving YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            manager = ConfigManager()
            manager.user_config_path = config_path
            
            config = AppConfig(
                audio=AudioConfig(sample_rate=22050),
                stt=STTConfig(),
                character=CharacterConfig(),
                tts=TTSConfig(),
                performance=PerformanceConfig(),
                logging=LoggingConfig()
            )
            
            manager.save_config(config, format="yaml")
            
            # Verify file was created and contains correct data
            assert config_path.exists()
            with open(config_path, 'r') as f:
                saved_data = yaml.safe_load(f)
            assert saved_data['audio']['sample_rate'] == 22050
    
    def test_save_json_config(self):
        """Test saving JSON configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            manager = ConfigManager()
            manager.user_config_path = config_path.with_suffix('.yaml')  # Will be changed to .json
            
            config = AppConfig(
                audio=AudioConfig(sample_rate=44100),
                stt=STTConfig(),
                character=CharacterConfig(),
                tts=TTSConfig(),
                performance=PerformanceConfig(),
                logging=LoggingConfig()
            )
            
            manager.save_config(config, format="json")
            
            # Verify file was created with .json extension
            json_path = config_path
            assert json_path.exists()
            with open(json_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data['audio']['sample_rate'] == 44100
    
    def test_validate_config(self):
        """Test configuration validation."""
        manager = ConfigManager()
        
        # Valid config should pass
        valid_config = AppConfig(
            audio=AudioConfig(),
            stt=STTConfig(),
            character=CharacterConfig(),
            tts=TTSConfig(),
            performance=PerformanceConfig(),
            logging=LoggingConfig()
        )
        assert manager.validate_config(valid_config) is True
        
        # Create invalid config by modifying after creation
        invalid_config = AppConfig(
            audio=AudioConfig(),
            stt=STTConfig(),
            character=CharacterConfig(),
            tts=TTSConfig(),
            performance=PerformanceConfig(),
            logging=LoggingConfig()
        )
        # Manually set invalid value to bypass validation
        invalid_config.audio.sample_rate = -1
        assert manager.validate_config(invalid_config) is False
    
    def test_get_config_loads_if_none(self):
        """Test get_config loads configuration if not already loaded."""
        with patch.object(ConfigManager, 'load_config') as mock_load:
            mock_config = AppConfig(
                audio=AudioConfig(),
                stt=STTConfig(),
                character=CharacterConfig(),
                tts=TTSConfig(),
                performance=PerformanceConfig(),
                logging=LoggingConfig()
            )
            mock_load.return_value = mock_config
            
            manager = ConfigManager()
            config = manager.get_config()
            
            mock_load.assert_called_once()
            assert config == mock_config
    
    def test_config_file_not_found_error(self):
        """Test error handling when config file not found."""
        manager = ConfigManager(Path("nonexistent/config.yaml"))
        
        with pytest.raises(ConfigValidationError, match="Configuration file not found"):
            manager.load_config()
    
    def test_invalid_yaml_format_error(self):
        """Test error handling for invalid YAML format."""
        with patch('builtins.open', mock_open(read_data="invalid: yaml: content: [")):
            with patch('pathlib.Path.exists', return_value=True):
                manager = ConfigManager()
                
                with pytest.raises(ConfigValidationError, match="Invalid configuration file format"):
                    manager.load_config()
    
    def test_invalid_json_format_error(self):
        """Test error handling for invalid JSON format."""
        with patch('builtins.open', mock_open(read_data='{"invalid": json}')):
            with patch('pathlib.Path.exists', return_value=True):
                manager = ConfigManager(Path("config/test.json"))
                
                with pytest.raises(ConfigValidationError, match="Invalid configuration file format"):
                    manager.load_config()
    
    def test_merge_configs(self):
        """Test configuration merging functionality."""
        manager = ConfigManager()
        
        base_config = {
            "audio": {"sample_rate": 16000, "chunk_size": 1024},
            "stt": {"model_size": "base"}
        }
        
        override_config = {
            "audio": {"sample_rate": 22050},  # Override sample_rate
            "tts": {"speed": 1.5}  # Add new section
        }
        
        merged = manager._merge_configs(base_config, override_config)
        
        assert merged["audio"]["sample_rate"] == 22050  # Overridden
        assert merged["audio"]["chunk_size"] == 1024    # Preserved
        assert merged["stt"]["model_size"] == "base"    # Preserved
        assert merged["tts"]["speed"] == 1.5            # Added


if __name__ == "__main__":
    pytest.main([__file__])