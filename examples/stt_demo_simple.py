"""
Simple STT demo with device selection - streamlined version.
"""

import asyncio
import numpy as np
import time
import pyaudio
from src.audio.device_manager import get_audio_devices, get_default_devices
from src.stt import WhisperSTT
from src.core.interfaces import PipelineConfig


def display_and_select_devices():
    """Display devices and get user selection."""
    print("=== Audio Device Selection ===\n")
    
    try:
        input_devices, output_devices = get_audio_devices()
        default_input, default_output = get_default_devices()
        
        print("Available Input Devices:")
        for i, device in enumerate(input_devices):
            default_marker = " (DEFAULT)" if default_input and device.id == default_input.id else ""
            print(f"  {i}: [{device.id}] {device.name}{default_marker}")
        
        print("\nAvailable Output Devices:")
        for i, device in enumerate(output_devices):
            default_marker = " (DEFAULT)" if default_output and device.id == default_output.id else ""
            print(f"  {i}: [{device.id}] {device.name}{default_marker}")
        
        # Get selections
        input_idx = int(input(f"\nSelect input device (0-{len(input_devices)-1}): "))
        output_idx = int(input(f"Select output device (0-{len(output_devices)-1}): "))
        
        selected_input = input_devices[input_idx]
        selected_output = output_devices[output_idx]
        
        print(f"\n✓ Selected Input: [{selected_input.id}] {selected_input.name}")
        print(f"✓ Selected Output: [{selected_output.id}] {selected_output.name}")
        
        return selected_input.id, selected_output.id
        
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0


def test_audio_playback(device_id: int):
    """Test audio playback."""
    print(f"\n🔊 Testing audio playback on device {device_id}...")
    
    # Generate test tone
    sample_rate = 16000
    duration = 1.5
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    fade_samples = int(0.1 * sample_rate)
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    audio = audio.astype(np.float32)
    
    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=device_id,
            frames_per_buffer=1024
        )
        
        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            stream.write(chunk.tobytes())
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print("✅ Audio playback successful!")
        
    except Exception as e:
        print(f"❌ Audio playback failed: {e}")


async def test_speech_recognition(input_device_id: int):
    """Test speech recognition with microphone."""
    print(f"\n🎤 Testing speech recognition with device {input_device_id}...")
    print("Speak for 3-5 seconds when prompted...")
    
    config = PipelineConfig(
        audio_device_id=input_device_id,
        sample_rate=16000,
        stt_model_size="tiny",
        gpu_device="cpu"
    )
    
    whisper_stt = WhisperSTT(config)
    
    try:
        print("🔄 Initializing Whisper...")
        await whisper_stt.initialize()
        
        input("\nPress Enter, then speak for 3-5 seconds...")
        
        # Record audio
        sample_rate = 16000
        record_seconds = 5
        chunk_size = 1024
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=input_device_id,
            frames_per_buffer=chunk_size
        )
        
        print("🎤 Recording... speak now!")
        
        frames = []
        num_chunks = int(sample_rate * record_seconds / chunk_size)
        
        for i in range(num_chunks):
            data = stream.read(chunk_size)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)
            
            # Show progress
            progress = (i + 1) / num_chunks
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"\r[{bar}] {progress:.0%}", end="", flush=True)
        
        print("\n🔄 Processing speech...")
        
        # Combine and process
        audio_data = np.concatenate(frames)
        
        start_time = time.time()
        text = await whisper_stt.transcribe(audio_data)
        processing_time = (time.time() - start_time) * 1000
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if text.strip():
            print(f"✅ Transcribed: '{text}'")
            print(f"⏱️  Processing time: {processing_time:.1f}ms")
        else:
            print("⚠️  No speech detected")
        
    except Exception as e:
        print(f"❌ Speech recognition failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await whisper_stt.cleanup()


async def main():
    """Main demo function."""
    print("=== STT Demo with Device Selection ===\n")
    
    try:
        # Select devices
        input_device_id, output_device_id = display_and_select_devices()
        
        # Test audio playback
        test_audio_playback(output_device_id)
        
        # Test speech recognition
        await test_speech_recognition(input_device_id)
        
        print("\n🎉 Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())