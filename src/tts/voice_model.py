"""
Voice model management for TTS system.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import soundfile as sf
import numpy as np

from ..core.interfaces import VoiceModel, PipelineError, PipelineStage


logger = logging.getLogger(__name__)


class VoiceModelManager:
    """
    Manages voice models for TTS synthesis including loading, validation, and caching.
    """
    
    def __init__(self, models_directory: str = "models/voices"):
        """
        Initialize voice model manager.
        
        Args:
            models_directory: Directory containing voice model files
        """
        self.models_directory = Path(models_directory)
        self.loaded_models: Dict[str, VoiceModel] = {}
        self.model_cache: Dict[str, Dict[str, Any]] = {}
        
        # Supported audio formats for voice cloning
        self.supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        # Default voice model configurations
        self.default_models = {
            "default_female": VoiceModel(
                name="default_female",
                model_path="",
                sample_rate=22050,
                language="en",
                gender="female"
            ),
            "default_male": VoiceModel(
                name="default_male",
                model_path="",
                sample_rate=22050,
                language="en",
                gender="male"
            )
        }
    
    def initialize(self) -> None:
        """Initialize the voice model manager and scan for available models."""
        try:
            # Create models directory if it doesn't exist
            self.models_directory.mkdir(parents=True, exist_ok=True)
            
            # Load default models
            self.loaded_models.update(self.default_models)
            
            # Scan for voice model files
            self._scan_voice_models()
            
            logger.info(f"Voice model manager initialized with {len(self.loaded_models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice model manager: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"Voice model manager initialization failed: {e}",
                recoverable=False
            )
    
    def _scan_voice_models(self) -> None:
        """Scan the models directory for voice model files."""
        try:
            if not self.models_directory.exists():
                logger.warning(f"Models directory does not exist: {self.models_directory}")
                return
            
            # Look for audio files that can be used for voice cloning
            for file_path in self.models_directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    try:
                        voice_model = self._create_voice_model_from_file(file_path)
                        self.loaded_models[voice_model.name] = voice_model
                        logger.debug(f"Loaded voice model: {voice_model.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load voice model from {file_path}: {e}")
                
                # Look for JSON configuration files
                elif file_path.suffix.lower() == '.json':
                    try:
                        voice_model = self._load_voice_model_from_config(file_path)
                        if voice_model:
                            self.loaded_models[voice_model.name] = voice_model
                            logger.debug(f"Loaded voice model from config: {voice_model.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load voice model config from {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Error scanning voice models: {e}")
    
    def _create_voice_model_from_file(self, file_path: Path) -> VoiceModel:
        """
        Create a voice model from an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            VoiceModel instance
        """
        # Validate audio file
        try:
            with sf.SoundFile(str(file_path)) as audio_file:
                sample_rate = audio_file.samplerate
                duration = len(audio_file) / sample_rate
                
                # Check minimum duration for voice cloning (at least 3 seconds)
                if duration < 3.0:
                    logger.warning(f"Audio file {file_path} is too short ({duration:.1f}s) for voice cloning")
                
        except Exception as e:
            raise ValueError(f"Invalid audio file {file_path}: {e}")
        
        # Extract metadata from filename and directory structure
        model_name = file_path.stem
        
        # Try to infer language and gender from filename or directory
        language = self._infer_language(file_path)
        gender = self._infer_gender(file_path)
        
        return VoiceModel(
            name=model_name,
            model_path=str(file_path),
            sample_rate=sample_rate,
            language=language,
            gender=gender
        )
    
    def _load_voice_model_from_config(self, config_path: Path) -> Optional[VoiceModel]:
        """
        Load voice model from JSON configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            VoiceModel instance or None if invalid
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['name', 'language']
            for field in required_fields:
                if field not in config:
                    logger.warning(f"Missing required field '{field}' in {config_path}")
                    return None
            
            # Resolve model path relative to config file
            model_path = config.get('model_path', '')
            if model_path and not os.path.isabs(model_path):
                model_path = str(config_path.parent / model_path)
            
            # Validate model file exists if specified
            if model_path and not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                model_path = ""  # Use as built-in voice
            
            return VoiceModel(
                name=config['name'],
                model_path=model_path,
                sample_rate=config.get('sample_rate', 22050),
                language=config['language'],
                gender=config.get('gender', 'neutral')
            )
            
        except Exception as e:
            logger.error(f"Failed to load voice model config from {config_path}: {e}")
            return None
    
    def _infer_language(self, file_path: Path) -> str:
        """
        Infer language from file path or name.
        
        Args:
            file_path: Path to voice model file
            
        Returns:
            Language code (default: 'en')
        """
        path_str = str(file_path).lower()
        
        # Language mapping - order matters, check more specific patterns first
        language_map = {
            'es': ['spanish', 'esp'],
            'fr': ['french', 'fra'],
            'de': ['german', 'deu'],
            'it': ['italian', 'ita'],
            'pt': ['portuguese', 'por'],
            'ru': ['russian', 'rus'],
            'ja': ['japanese', 'jpn'],
            'ko': ['korean', 'kor'],
            'zh': ['chinese', 'chn', 'mandarin'],
            'en': ['english', 'eng']  # Check English last since 'en' is common
        }
        
        # First check for language codes at word boundaries or in directory names
        for lang_code, keywords in language_map.items():
            for keyword in keywords:
                if keyword in path_str:
                    return lang_code
        
        # Check for standalone language codes
        for lang_code in ['es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'en']:
            if f'_{lang_code}_' in path_str or f'/{lang_code}/' in path_str or f'\\{lang_code}\\' in path_str:
                return lang_code
        
        return 'en'  # Default to English
    
    def _infer_gender(self, file_path: Path) -> str:
        """
        Infer gender from file path or name.
        
        Args:
            file_path: Path to voice model file
            
        Returns:
            Gender ('male', 'female', or 'neutral')
        """
        path_str = str(file_path).lower()
        
        # Gender keywords
        if any(keyword in path_str for keyword in ['female', 'woman', 'girl', 'lady']):
            return 'female'
        elif any(keyword in path_str for keyword in ['male', 'man', 'boy', 'gentleman']):
            return 'male'
        else:
            return 'neutral'
    
    def get_voice_model(self, name: str) -> Optional[VoiceModel]:
        """
        Get voice model by name.
        
        Args:
            name: Voice model name
            
        Returns:
            VoiceModel instance or None if not found
        """
        return self.loaded_models.get(name)
    
    def get_available_voices(self) -> List[VoiceModel]:
        """
        Get list of all available voice models.
        
        Returns:
            List of VoiceModel instances
        """
        return list(self.loaded_models.values())
    
    def get_voices_by_language(self, language: str) -> List[VoiceModel]:
        """
        Get voice models filtered by language.
        
        Args:
            language: Language code (e.g., 'en', 'es')
            
        Returns:
            List of VoiceModel instances for the specified language
        """
        return [model for model in self.loaded_models.values() if model.language == language]
    
    def get_voices_by_gender(self, gender: str) -> List[VoiceModel]:
        """
        Get voice models filtered by gender.
        
        Args:
            gender: Gender ('male', 'female', 'neutral')
            
        Returns:
            List of VoiceModel instances for the specified gender
        """
        return [model for model in self.loaded_models.values() if model.gender == gender]
    
    def add_voice_model(self, voice_model: VoiceModel) -> None:
        """
        Add a voice model to the manager.
        
        Args:
            voice_model: VoiceModel instance to add
        """
        if not isinstance(voice_model, VoiceModel):
            raise ValueError("voice_model must be a VoiceModel instance")
        
        # Validate model if it has a file path
        if voice_model.model_path and not os.path.exists(voice_model.model_path):
            raise FileNotFoundError(f"Voice model file not found: {voice_model.model_path}")
        
        self.loaded_models[voice_model.name] = voice_model
        logger.info(f"Added voice model: {voice_model.name}")
    
    def remove_voice_model(self, name: str) -> bool:
        """
        Remove a voice model from the manager.
        
        Args:
            name: Voice model name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.loaded_models:
            del self.loaded_models[name]
            if name in self.model_cache:
                del self.model_cache[name]
            logger.info(f"Removed voice model: {name}")
            return True
        return False
    
    def validate_voice_model(self, voice_model: VoiceModel) -> bool:
        """
        Validate a voice model.
        
        Args:
            voice_model: VoiceModel to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not voice_model.name or not voice_model.language:
                return False
            
            # Check sample rate is reasonable
            if voice_model.sample_rate <= 0 or voice_model.sample_rate > 96000:
                return False
            
            # Check gender is valid
            valid_genders = {'male', 'female', 'neutral'}
            if voice_model.gender not in valid_genders:
                return False
            
            # If model has a file path, validate the file
            if voice_model.model_path:
                if not os.path.exists(voice_model.model_path):
                    return False
                
                # Try to read audio file
                try:
                    with sf.SoundFile(voice_model.model_path) as audio_file:
                        # Check if file is readable
                        if len(audio_file) == 0:
                            return False
                except Exception:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating voice model {voice_model.name}: {e}")
            return False
    
    def get_fallback_voice(self, language: str = "en", gender: str = "neutral") -> VoiceModel:
        """
        Get a fallback voice model when the requested one is not available.
        
        Args:
            language: Preferred language
            gender: Preferred gender
            
        Returns:
            Fallback VoiceModel
        """
        # Try to find a voice with matching language and gender
        candidates = [
            model for model in self.loaded_models.values()
            if model.language == language and model.gender == gender
        ]
        
        if candidates:
            return candidates[0]
        
        # Try to find a voice with matching language
        candidates = [
            model for model in self.loaded_models.values()
            if model.language == language
        ]
        
        if candidates:
            return candidates[0]
        
        # Try to find a voice with matching gender
        candidates = [
            model for model in self.loaded_models.values()
            if model.gender == gender
        ]
        
        if candidates:
            return candidates[0]
        
        # Return any available voice
        if self.loaded_models:
            return next(iter(self.loaded_models.values()))
        
        # Return default voice as last resort
        return self.default_models["default_female"]
    
    def save_voice_model_config(self, voice_model: VoiceModel, config_path: Optional[Path] = None) -> Path:
        """
        Save voice model configuration to JSON file.
        
        Args:
            voice_model: VoiceModel to save
            config_path: Optional path for config file
            
        Returns:
            Path to saved config file
        """
        if config_path is None:
            config_path = self.models_directory / f"{voice_model.name}.json"
        
        config = {
            "name": voice_model.name,
            "model_path": voice_model.model_path,
            "sample_rate": voice_model.sample_rate,
            "language": voice_model.language,
            "gender": voice_model.gender
        }
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved voice model config: {config_path}")
            return config_path
            
        except Exception as e:
            logger.error(f"Failed to save voice model config: {e}")
            raise PipelineError(
                PipelineStage.TEXT_TO_SPEECH,
                f"Failed to save voice model config: {e}",
                recoverable=True
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded voice models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "total_models": len(self.loaded_models),
            "models_directory": str(self.models_directory),
            "supported_formats": list(self.supported_formats),
            "models": {}
        }
        
        # Group models by language and gender
        by_language = {}
        by_gender = {}
        
        for model in self.loaded_models.values():
            # By language
            if model.language not in by_language:
                by_language[model.language] = []
            by_language[model.language].append(model.name)
            
            # By gender
            if model.gender not in by_gender:
                by_gender[model.gender] = []
            by_gender[model.gender].append(model.name)
            
            # Individual model info
            info["models"][model.name] = {
                "language": model.language,
                "gender": model.gender,
                "sample_rate": model.sample_rate,
                "has_file": bool(model.model_path and os.path.exists(model.model_path))
            }
        
        info["by_language"] = by_language
        info["by_gender"] = by_gender
        
        return info