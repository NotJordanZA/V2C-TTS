"""
Integration tests for configuration management system.
"""

import pytest
from pathlib import Path
from src.core.config import ConfigManager, AppConfig


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_load_default_config_file(self):
        """Test loading the actual default configuration file."""
        # Use the actual default config file
        config_path = Path("config/default_config.yaml")
        
        if not config_path.exists():
            pytest.skip("Default config file not found")
        
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        # Verify the config loaded successfully
        assert isinstance(config, AppConfig)
        assert config.audio.sample_rate > 0
        assert config.stt.model_size in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        assert config.character.llm_model_path != ""
        assert config.tts.model_path != ""
    
    def test_pipeline_config_conversion(self):
        """Test conversion to PipelineConfig with real config."""
        config_path = Path("config/default_config.yaml")
        
        if not config_path.exists():
            pytest.skip("Default config file not found")
        
        manager = ConfigManager(config_path)
        app_config = manager.load_config()
        pipeline_config = app_config.to_pipeline_config()
        
        # Verify pipeline config has expected values
        assert pipeline_config.sample_rate == app_config.audio.sample_rate
        assert pipeline_config.chunk_size == app_config.audio.chunk_size
        assert pipeline_config.stt_model_size == app_config.stt.model_size
        assert pipeline_config.max_latency_ms == app_config.performance.max_latency_ms


if __name__ == "__main__":
    pytest.main([__file__])