"""
Configuration management for the voice transformation system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from .interfaces import PipelineConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class AudioConfig:
    """Audio configuration settings."""
    sample_rate: int = 16000
    chunk_size: int = 1024
    input_device_id: int = -1
    output_device_id: int = -1
    vad_threshold: float = 0.5
    
    def __post_init__(self):
        """Validate audio configuration."""
        if self.sample_rate <= 0:
            raise ConfigValidationError("sample_rate must be positive")
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ConfigValidationError(f"sample_rate {self.sample_rate} not supported")
        if self.chunk_size <= 0 or self.chunk_size > 8192:
            raise ConfigValidationError("chunk_size must be between 1 and 8192")
        if not (0.0 <= self.vad_threshold <= 1.0):
            raise ConfigValidationError("vad_threshold must be between 0.0 and 1.0")


@dataclass
class STTConfig:
    """Speech-to-Text configuration settings."""
    model_size: str = "base"
    device: str = "cuda"
    language: str = "auto"
    
    def __post_init__(self):
        """Validate STT configuration."""
        valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if self.model_size not in valid_sizes:
            raise ConfigValidationError(f"model_size must be one of {valid_sizes}")
        if self.device not in ["cuda", "cpu"]:
            raise ConfigValidationError("device must be 'cuda' or 'cpu'")


@dataclass
class CharacterConfig:
    """Character transformation configuration settings."""
    default_character: str = "neutral"
    intensity: float = 1.0
    llm_model_path: str = "models/llama-3.1-8b-instruct.gguf"
    max_tokens: int = 256
    temperature: float = 0.7
    
    def __post_init__(self):
        """Validate character configuration."""
        if not (0.0 <= self.intensity <= 2.0):
            raise ConfigValidationError("intensity must be between 0.0 and 2.0")
        if self.max_tokens <= 0 or self.max_tokens > 4096:
            raise ConfigValidationError("max_tokens must be between 1 and 4096")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigValidationError("temperature must be between 0.0 and 2.0")
        if not self.llm_model_path:
            raise ConfigValidationError("llm_model_path cannot be empty")


@dataclass
class TTSConfig:
    """Text-to-Speech configuration settings."""
    model_path: str = "models/tts_model.pth"
    device: str = "cuda"
    sample_rate: int = 22050
    speed: float = 1.0
    
    def __post_init__(self):
        """Validate TTS configuration."""
        if self.device not in ["cuda", "cpu"]:
            raise ConfigValidationError("device must be 'cuda' or 'cpu'")
        if self.sample_rate not in [16000, 22050, 44100, 48000]:
            raise ConfigValidationError(f"sample_rate {self.sample_rate} not supported for TTS")
        if not (0.1 <= self.speed <= 3.0):
            raise ConfigValidationError("speed must be between 0.1 and 3.0")
        if not self.model_path:
            raise ConfigValidationError("model_path cannot be empty")


@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    max_latency_ms: int = 2000
    gpu_memory_fraction: float = 0.8
    enable_model_offloading: bool = True
    batch_size: int = 1
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.max_latency_ms <= 0 or self.max_latency_ms > 10000:
            raise ConfigValidationError("max_latency_ms must be between 1 and 10000")
        if not (0.1 <= self.gpu_memory_fraction <= 1.0):
            raise ConfigValidationError("gpu_memory_fraction must be between 0.1 and 1.0")
        if self.batch_size <= 0 or self.batch_size > 32:
            raise ConfigValidationError("batch_size must be between 1 and 32")


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    file: str = "logs/voice_transform.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ConfigValidationError(f"level must be one of {valid_levels}")
        if self.backup_count < 0 or self.backup_count > 100:
            raise ConfigValidationError("backup_count must be between 0 and 100")
        if not self.file:
            raise ConfigValidationError("file path cannot be empty")
        # Validate max_file_size format (e.g., "10MB", "1GB")
        import re
        if not re.match(r'^\d+[KMGT]?B$', self.max_file_size):
            raise ConfigValidationError("max_file_size must be in format like '10MB', '1GB'")


@dataclass
class AppConfig:
    """Main application configuration."""
    audio: AudioConfig
    stt: STTConfig
    character: CharacterConfig
    tts: TTSConfig
    performance: PerformanceConfig
    logging: LoggingConfig

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to PipelineConfig for pipeline components."""
        return PipelineConfig(
            audio_device_id=self.audio.input_device_id,
            sample_rate=self.audio.sample_rate,
            chunk_size=self.audio.chunk_size,
            stt_model_size=self.stt.model_size,
            llm_model_path=self.character.llm_model_path,
            tts_model_path=self.tts.model_path,
            gpu_device=self.stt.device,
            max_latency_ms=self.performance.max_latency_ms
        )


class ConfigManager:
    """Manages application configuration loading and saving."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/default_config.yaml")
        self.user_config_path = Path("config/user_config.yaml")
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from files."""
        try:
            # Start with default config
            default_config = self._load_config_file(self.config_path)
            
            # Override with user config if it exists
            if self.user_config_path.exists():
                user_config = self._load_config_file(self.user_config_path)
                default_config = self._merge_configs(default_config, user_config)
            
            # Convert to dataclass and validate
            self._config = self._dict_to_config(default_config)
            return self._config
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration: {e}")
    
    def save_config(self, config: AppConfig, format: str = "yaml") -> None:
        """Save configuration to user config file."""
        try:
            self.user_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update file extension based on format
            if format.lower() == "json":
                self.user_config_path = self.user_config_path.with_suffix('.json')
            else:
                self.user_config_path = self.user_config_path.with_suffix('.yaml')
            
            config_dict = asdict(config)
            
            if format.lower() == "json":
                with open(self.user_config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                with open(self.user_config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self._config = config
        except Exception as e:
            raise ConfigValidationError(f"Failed to save configuration: {e}")
    
    def get_config(self) -> AppConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def _load_config_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration file (YAML or JSON)."""
        if not path.exists():
            raise ConfigValidationError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigValidationError(f"Invalid configuration file format: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to read configuration file: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig dataclass with validation."""
        try:
            return AppConfig(
                audio=AudioConfig(**config_dict.get('audio', {})),
                stt=STTConfig(**config_dict.get('stt', {})),
                character=CharacterConfig(**config_dict.get('character', {})),
                tts=TTSConfig(**config_dict.get('tts', {})),
                performance=PerformanceConfig(**config_dict.get('performance', {})),
                logging=LoggingConfig(**config_dict.get('logging', {}))
            )
        except TypeError as e:
            raise ConfigValidationError(f"Invalid configuration structure: {e}")
    
    def validate_config(self, config: AppConfig) -> bool:
        """Validate configuration without saving."""
        try:
            # Trigger validation by converting to dict and back
            config_dict = asdict(config)
            self._dict_to_config(config_dict)
            return True
        except ConfigValidationError:
            return False