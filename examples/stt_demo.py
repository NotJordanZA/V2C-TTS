"""
Demo script showing STT module functionality with device selection.
"""

import asyncio
import numpy as np
import time
import logging

from src.core.interfaces import PipelineConfig, AudioChunk
from src.stt import STTProcessor, STTResult
from src.audio.device_manager import AudioDeviceManager, get_audio_devices, get_default_devices
from src.audio.capture import AudioCapture
from src.audio.output import AudioOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_audio(text_description: str, duration: float = 2.0) -> AudioChunk:
    """Create sample audio data for testing."""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create more realistic speech-like audio that won't be filtered by VAD
    # Use multiple harmonics and modulation to simulate speech patterns
    base_freq = 200  # Base frequency around human speech
    
    if "hello" in text_description.lower():
        # Simulate "hello" with rising then falling pitch
        freq_mod = base_freq + 100 * np.sin(2 * np.pi * 2 * t)  # Pitch variation
        audio = np.sin(2 * np.pi * freq_mod * t) * 0.4
        # Add harmonics
        audio += 0.2 * np.sin(2 * np.pi * freq_mod * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * freq_mod * 3 * t)
    elif "world" in text_description.lower():
        # Simulate "world" with lower, more stable pitch
        freq_mod = base_freq + 50 * np.sin(2 * np.pi * 1.5 * t)
        audio = np.sin(2 * np.pi * freq_mod * t) * 0.4
        audio += 0.2 * np.sin(2 * np.pi * freq_mod * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * freq_mod * 3 * t)
    else:
        # Mixed speech-like patterns
        freq_mod = base_freq + 80 * np.sin(2 * np.pi * 1.8 * t)
        audio = np.sin(2 * np.pi * freq_mod * t) * 0.4
        audio += 0.2 * np.sin(2 * np.pi * freq_mod * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * freq_mod * 3 * t)
        # Add some formant-like resonances
        audio += 0.15 * np.sin(2 * np.pi * 800 * t) * np.exp(-t * 2)
        audio += 0.1 * np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 3)
    
    # Add amplitude modulation to simulate speech envelope
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))  # 5 Hz modulation
    audio = audio * envelope
    
    # Add realistic noise level
    noise = np.random.randn(len(audio)) * 0.02  # Lower noise level
    audio = (audio + noise).astype(np.float32)
    
    # Ensure the audio has sufficient energy to pass VAD
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.1:  # Boost if too quiet
        audio = audio * (0.1 / rms)
    
    return AudioChunk(
        data=audio,
        timestamp=time.time(),
        sample_rate=sample_rate,
        duration_ms=duration * 1000
    )


def display_devices():
    """Display available audio devices and let user select."""
    print("=== Audio Device Selection ===\n")
    
    try:
        input_devices, output_devices = get_audio_devices()
        default_input, default_output = get_default_devices()
        
        print("Available Input Devices:")
        for i, device in enumerate(input_devices):
            default_marker = " (DEFAULT)" if default_input and device.id == default_input.id else ""
            print(f"  {i}: [{device.id}] {device.name} - {device.channels} channels, {device.sample_rate}Hz{default_marker}")
        
        print("\nAvailable Output Devices:")
        for i, device in enumerate(output_devices):
            default_marker = " (DEFAULT)" if default_output and device.id == default_output.id else ""
            print(f"  {i}: [{device.id}] {device.name} - {device.channels} channels, {device.sample_rate}Hz{default_marker}")
        
        # Get user selection for input device
        if input_devices:
            while True:
                try:
                    choice = input(f"\nSelect input device (0-{len(input_devices)-1}, or press Enter for default): ").strip()
                    if choice == "":
                        selected_input = default_input.id if default_input else input_devices[0].id
                        selected_input_name = default_input.name if default_input else input_devices[0].name
                        break
                    else:
                        idx = int(choice)
                        if 0 <= idx < len(input_devices):
                            selected_input = input_devices[idx].id
                            selected_input_name = input_devices[idx].name
                            break
                        else:
                            print(f"Please enter a number between 0 and {len(input_devices)-1}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            selected_input = 0
            selected_input_name = "Default"
        
        # Get user selection for output device
        if output_devices:
            while True:
                try:
                    choice = input(f"Select output device (0-{len(output_devices)-1}, or press Enter for default): ").strip()
                    if choice == "":
                        selected_output = default_output.id if default_output else output_devices[0].id
                        selected_output_name = default_output.name if default_output else output_devices[0].name
                        break
                    else:
                        idx = int(choice)
                        if 0 <= idx < len(output_devices):
                            selected_output = output_devices[idx].id
                            selected_output_name = output_devices[idx].name
                            break
                        else:
                            print(f"Please enter a number between 0 and {len(output_devices)-1}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            selected_output = 0
            selected_output_name = "Default"
        
        print(f"\nSelected Input Device: [{selected_input}] {selected_input_name}")
        print(f"Selected Output Device: [{selected_output}] {selected_output_name}")
        
        return selected_input, selected_output, selected_input_name, selected_output_name
        
    except Exception as e:
        print(f"Error accessing audio devices: {e}")
        print("Using default device IDs (0)")
        return 0, 0, "Default", "Default"


async def stt_demo_with_devices(input_device_id: int, output_device_id: int, input_name: str, output_name: str):
    """Demonstrate STT functionality with selected devices."""
    print("=== Synthetic Audio STT Demo ===\n")
    
    # Create configuration with selected devices
    config = PipelineConfig(
        audio_device_id=input_device_id,
        sample_rate=16000,
        stt_model_size="tiny",  # Use smallest model for demo
        gpu_device="cpu"  # Use CPU for compatibility
    )
    
    print(f"Using Input Device: {input_name}")
    print(f"Using Output Device: {output_name}")
    
    # Results storage
    results = []
    
    def result_callback(result: STTResult):
        """Handle STT results."""
        results.append(result)
        print(f"✓ Transcribed: '{result.text}' "
              f"(processing time: {result.processing_time_ms:.1f}ms)")
    
    # Create STT processor
    processor = STTProcessor(config, result_callback)
    
    try:
        print("1. Initializing STT processor...")
        await processor.initialize()
        print(f"   Model info: {processor.whisper_stt.get_model_info()}")
        
        print("\n2. Starting processing pipeline...")
        await processor.start_processing()
        
        print("\n3. Processing sample audio chunks...")
        
        # Create sample audio chunks
        test_samples = [
            ("hello", 1.5),
            ("world", 1.0),
            ("testing speech recognition", 2.0),
            ("short", 0.8)
        ]
        
        for i, (description, duration) in enumerate(test_samples, 1):
            print(f"\n   Processing sample {i}: '{description}'")
            
            # Create and process audio chunk
            audio_chunk = create_sample_audio(description, duration)
            await processor.process_audio(audio_chunk)
            
            # Wait a bit for processing
            await asyncio.sleep(0.5)
        
        # Wait for all processing to complete
        print("\n4. Waiting for processing to complete...")
        await asyncio.sleep(2.0)
        
        # Show metrics
        print("\n5. Processing metrics:")
        metrics = processor.get_metrics()
        for key, value in metrics.items():
            if key != "whisper_model_info":
                print(f"   {key}: {value}")
        
        print(f"\n6. Total results received: {len(results)}")
        
        # Test audio playback with selected output device
        print("\n7. Testing audio playback...")
        print("   Playing a test tone through selected output device...")
        try:
            # Use direct PyAudio for simpler, more reliable playback
            import pyaudio
            
            # Generate a simple test tone
            sample_rate = 16000
            duration = 1.0
            frequency = 440  # A4 note
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create test tone with fade in/out
            test_audio = np.sin(2 * np.pi * frequency * t) * 0.3
            fade_samples = int(0.1 * sample_rate)
            test_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            test_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            test_audio = test_audio.astype(np.float32)
            
            # Play using PyAudio directly
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=output_device_id,
                frames_per_buffer=1024
            )
            
            # Play in chunks
            chunk_size = 1024
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                stream.write(chunk.tobytes())
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            print("   ✓ Audio playback completed successfully!")
            
        except Exception as e:
            print(f"   ✗ Audio playback failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test preprocessing configuration
        print("\n8. Testing preprocessing configuration...")
        processor.configure_preprocessing(
            min_audio_length_ms=500,
            silence_threshold=0.02
        )
        
        # Test with very short audio (should be filtered)
        short_audio = create_sample_audio("too short", 0.3)
        print("   Processing very short audio (should be filtered)...")
        await processor.process_audio(short_audio)
        await asyncio.sleep(0.5)
        
        # Test with silence (should be filtered)
        silence_chunk = AudioChunk(
            data=np.zeros(16000, dtype=np.float32),
            timestamp=time.time(),
            sample_rate=16000,
            duration_ms=1000
        )
        print("   Processing silence (should be filtered)...")
        await processor.process_audio(silence_chunk)
        await asyncio.sleep(0.5)
        
        print(f"\n   Final results count: {len(results)} (should be same as before)")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        print("\n9. Cleaning up...")
        await processor.cleanup()
        print("   Cleanup complete!")
    
    print("\n=== Demo Complete ===")


async def real_time_stt_demo(input_device_id: int, output_device_id: int):
    """Demonstrate real-time STT with actual microphone input."""
    print("\n=== Real-Time STT Demo ===\n")
    print("This demo will capture audio from your microphone and transcribe it in real-time.")
    print("Speak for 3-5 seconds, then pause. The system will process your speech.")
    print("Press Enter to start, then speak into your microphone. Press Ctrl+C to stop.\n")
    
    input("Press Enter to continue...")
    
    from src.stt import WhisperSTT
    import pyaudio
    import threading
    
    config = PipelineConfig(
        audio_device_id=input_device_id,
        sample_rate=16000,
        stt_model_size="tiny",
        gpu_device="cpu"
    )
    
    # Initialize Whisper STT
    whisper_stt = WhisperSTT(config)
    
    try:
        print("1. Initializing Whisper STT...")
        await whisper_stt.initialize()
        
        print("2. Starting real-time capture and transcription...")
        print("   Speak into your microphone. Speak for 3-5 seconds, then pause.")
        print("   The system will process your speech during pauses.")
        print("   Press Ctrl+C to stop.\n")
        
        # Audio capture setup
        sample_rate = 16000
        chunk_size = 1024
        record_seconds = 5  # Record in 5-second chunks
        
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=input_device_id,
            frames_per_buffer=chunk_size
        )
        
        print("🎤 Recording started. Speak now...")
        
        transcription_count = 0
        
        try:
            while True:
                # Record for a few seconds
                print(f"\n--- Recording chunk {transcription_count + 1} ---")
                frames = []
                num_chunks = int(sample_rate * record_seconds / chunk_size)
                
                for i in range(num_chunks):
                    data = stream.read(chunk_size)
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    frames.append(audio_chunk)
                    
                    # Show simple level meter
                    rms = np.sqrt(np.mean(audio_chunk ** 2))
                    level_bars = int(rms * 50)
                    level_display = "█" * min(level_bars, 20) + "░" * (20 - min(level_bars, 20))
                    print(f"\rLevel: [{level_display}] {rms:.3f}", end="", flush=True)
                
                # Combine all frames
                audio_data = np.concatenate(frames)
                
                # Check if there's significant audio
                rms_level = np.sqrt(np.mean(audio_data ** 2))
                if rms_level > 0.01:  # Only process if there's some audio
                    print(f"\n🔄 Processing audio (RMS: {rms_level:.3f})...")
                    
                    try:
                        # Transcribe
                        start_time = time.time()
                        text = await whisper_stt.transcribe(audio_data)
                        processing_time = (time.time() - start_time) * 1000
                        
                        if text.strip():
                            transcription_count += 1
                            print(f"🎤 Transcribed: '{text}' ({processing_time:.1f}ms)")
                        else:
                            print("   (No speech detected)")
                    
                    except Exception as e:
                        print(f"   ❌ Transcription failed: {e}")
                else:
                    print(f"\n   (Audio too quiet, RMS: {rms_level:.3f})")
                
                print("   Speak again or press Ctrl+C to stop...")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping capture...")
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        
        print(f"\n✅ Session complete! Total transcriptions: {transcription_count}")
        
    except Exception as e:
        print(f"❌ Real-time demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🧹 Cleaning up...")
        await whisper_stt.cleanup()
        print("   Cleanup complete!")
    
    print("\n=== Real-Time Demo Complete ===")


async def whisper_direct_demo():
    """Demonstrate direct WhisperSTT usage."""
    print("\n=== Direct WhisperSTT Demo ===\n")
    
    from src.stt import WhisperSTT
    
    config = PipelineConfig(
        audio_device_id=0,
        stt_model_size="tiny",
        gpu_device="cpu"
    )
    
    whisper = WhisperSTT(config)
    
    try:
        print("1. Initializing Whisper model...")
        await whisper.initialize()
        
        print("2. Testing direct transcription...")
        
        # Create sample audio
        sample_audio = create_sample_audio("direct test", 1.5)
        
        # Transcribe directly
        start_time = time.time()
        text = await whisper.transcribe(sample_audio.data)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"   Result: '{text}'")
        print(f"   Processing time: {processing_time:.1f}ms")
        
        print("3. Testing model info...")
        info = whisper.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
    finally:
        print("\n4. Cleaning up...")
        await whisper.cleanup()
        print("   Cleanup complete!")
    
    print("\n=== Direct Demo Complete ===")


if __name__ == "__main__":
    async def main():
        try:
            # Get device selection once for all demos
            input_device_id, output_device_id, input_name, output_name = display_devices()
            
            print("\n" + "="*60)
            print("STT Demo Options:")
            print("1. Synthetic Audio Demo (generated test audio)")
            print("2. Real-Time Microphone Demo (live audio capture)")
            print("3. Direct WhisperSTT Demo (low-level API)")
            print("4. All Demos")
            print("="*60)
            
            while True:
                try:
                    choice = input("\nSelect demo (1-4): ").strip()
                    if choice in ["1", "2", "3", "4"]:
                        break
                    else:
                        print("Please enter 1, 2, 3, or 4")
                except KeyboardInterrupt:
                    print("\nDemo cancelled by user")
                    return
            
            if choice in ["1", "4"]:
                # Update the config in stt_demo to use selected devices
                await stt_demo_with_devices(input_device_id, output_device_id, input_name, output_name)
            
            if choice in ["2", "4"]:
                await real_time_stt_demo(input_device_id, output_device_id)
            
            if choice in ["3", "4"]:
                await whisper_direct_demo()
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"\nDemo failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the demo
    asyncio.run(main())