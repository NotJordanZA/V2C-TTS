"""
Demo script showing audio capture and output functionality.
This example captures audio from the microphone and plays it back through speakers
with a small delay to demonstrate the audio pipeline.
"""

import asyncio
import numpy as np
import logging
import signal
import sys
import os

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audio import AudioCapture, AudioOutput
from src.core.interfaces import PipelineConfig, AudioChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioLoopbackDemo:
    """Demo class that captures audio and plays it back with processing."""
    
    def __init__(self):
        # Create configuration
        self.config = PipelineConfig(
            audio_device_id=0,  # Will be set dynamically
            sample_rate=16000,
            chunk_size=1024
        )
        
        # Create audio components
        self.audio_capture = AudioCapture(self.config)
        self.audio_output = AudioOutput(self.config)
        
        # Demo state
        self.running = False
        self.processed_chunks = 0
    
    async def initialize(self):
        """Initialize audio components."""
        logger.info("Initializing audio components...")
        
        await self.audio_capture.initialize()
        await self.audio_output.initialize()
        
        # List available devices
        input_devices = self.audio_capture.get_available_devices()
        output_devices = self.audio_output.get_available_devices()
        
        logger.info(f"Found {len(input_devices)} input devices:")
        for device in input_devices:
            logger.info(f"  {device.id}: {device.name} ({device.channels} channels, {device.sample_rate} Hz)")
        
        logger.info(f"Found {len(output_devices)} output devices:")
        for device in output_devices:
            logger.info(f"  {device.id}: {device.name} ({device.channels} channels, {device.sample_rate} Hz)")
        
        # Use default devices or first available
        input_device_id = input_devices[0].id if input_devices else 0
        
        logger.info(f"Using input device: {input_device_id}")
        
        return input_device_id
    
    async def cleanup(self):
        """Clean up audio components."""
        logger.info("Cleaning up audio components...")
        await self.audio_capture.cleanup()
        await self.audio_output.cleanup()
    
    def audio_callback(self, audio_chunk: AudioChunk):
        """Process captured audio chunks."""
        self.processed_chunks += 1
        
        logger.info(f"Processed chunk {self.processed_chunks}: "
                   f"{len(audio_chunk.data)} samples, "
                   f"{audio_chunk.duration_ms:.1f}ms duration")
        
        # Simple processing: apply a small gain and add some reverb effect
        processed_audio = self.process_audio(audio_chunk.data)
        
        # Play back the processed audio
        asyncio.create_task(self.audio_output.play_audio(processed_audio, audio_chunk.sample_rate))
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simple audio processing effects."""
        # Apply gain (make it quieter to avoid feedback)
        processed = audio_data * 0.3
        
        # Add simple delay effect (very basic reverb)
        if len(processed) > 800:  # Only if we have enough samples
            delay_samples = 800  # ~50ms delay at 16kHz
            delayed = np.zeros_like(processed)
            delayed[delay_samples:] = processed[:-delay_samples] * 0.2
            processed = processed + delayed
        
        return processed
    
    async def run_demo(self, duration_seconds: int = 30):
        """Run the audio loopback demo."""
        try:
            # Initialize
            input_device_id = await self.initialize()
            
            # Set up signal handler for graceful shutdown
            def signal_handler(signum, frame):
                logger.info("Received interrupt signal, stopping demo...")
                self.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            
            # Start audio capture
            logger.info(f"Starting audio capture for {duration_seconds} seconds...")
            logger.info("Speak into your microphone - you should hear your voice played back with effects!")
            logger.info("Press Ctrl+C to stop early.")
            
            await self.audio_capture.start_capture(input_device_id, self.audio_callback)
            self.running = True
            
            # Run for specified duration or until interrupted
            start_time = asyncio.get_event_loop().time()
            while self.running and (asyncio.get_event_loop().time() - start_time) < duration_seconds:
                await asyncio.sleep(0.1)
                
                # Print status every 5 seconds
                elapsed = asyncio.get_event_loop().time() - start_time
                if int(elapsed) % 5 == 0 and int(elapsed * 10) % 50 == 0:  # Every 5 seconds
                    capture_status = self.audio_capture.get_buffer_status()
                    playback_status = self.audio_output.get_playback_status()
                    
                    logger.info(f"Status after {int(elapsed)}s: "
                               f"Captured {self.processed_chunks} chunks, "
                               f"Buffer: {capture_status['buffer_duration_seconds']:.2f}s, "
                               f"Playing: {playback_status['is_playing']}")
            
            logger.info(f"Demo completed. Processed {self.processed_chunks} audio chunks.")
            
        except Exception as e:
            logger.error(f"Error during demo: {e}")
            raise
        finally:
            # Clean up
            await self.audio_capture.stop_capture()
            await self.audio_output.stop_playback()
            await self.cleanup()


async def main():
    """Main demo function."""
    print("Audio Capture and Output Demo")
    print("=" * 40)
    print("This demo will capture audio from your microphone and play it back")
    print("through your speakers with some simple audio effects applied.")
    print("Make sure your microphone and speakers are working!")
    print()
    
    # Check if user wants to run the demo
    try:
        response = input("Do you want to run the demo? (y/n): ").lower().strip()
        if response != 'y':
            print("Demo cancelled.")
            return
        
        duration = input("Enter duration in seconds (default 10): ").strip()
        try:
            duration = int(duration) if duration else 10
        except ValueError:
            duration = 10
        
        print(f"Running demo for {duration} seconds...")
        print()
        
        # Run the demo
        demo = AudioLoopbackDemo()
        await demo.run_demo(duration)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)
    
    print("Demo finished successfully!")


if __name__ == "__main__":
    asyncio.run(main())