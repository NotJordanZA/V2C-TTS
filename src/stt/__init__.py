"""
Speech-to-Text module for voice transformation pipeline.
"""

from .whisper_stt import WhisperSTT
from .processor import STTProcessor, STTResult, ProcessingState

__all__ = ["WhisperSTT", "STTProcessor", "STTResult", "ProcessingState"]