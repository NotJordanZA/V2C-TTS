"""
Simple STT test to verify functionality.
"""

import asyncio
import numpy as np
import time
from src.core.interfaces import PipelineConfig, AudioChunk
from src.stt import STTProcessor, STTResult


async def simple_test():
    """Simple test of STT functionality."""
    print("=== Simple STT Test ===")
    
    config = PipelineConfig(
        audio_device_id=0,
        sample_rate=16000,
        stt_model_size="tiny",
        gpu_device="cpu"
    )
    
    results = []
    def callback(result: STTResult):
        results.append(result)
        print(f"Transcribed: '{result.text}' ({result.processing_time_ms:.1f}ms)")
    
    processor = STTProcessor(config, callback)
    
    try:
        await processor.initialize()
        await processor.start_processing()
        
        # Create a simple sine wave that should be processed
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a more speech-like signal
        audio = np.sin(2 * np.pi * 300 * t) * 0.3  # Base tone
        audio += np.sin(2 * np.pi * 600 * t) * 0.2  # Harmonic
        audio += np.random.randn(len(audio)) * 0.05  # Noise
        
        chunk = AudioChunk(
            data=audio.astype(np.float32),
            timestamp=time.time(),
            sample_rate=sample_rate,
            duration_ms=duration * 1000
        )
        
        print("Processing audio chunk...")
        await processor.process_audio(chunk)
        
        # Wait for processing
        await asyncio.sleep(3)
        
        print(f"Results: {len(results)}")
        
    finally:
        await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(simple_test())