"""
Simple audio input/output test to verify device functionality.
"""

import pyaudio
import numpy as np
import time
import threading
from src.audio.device_manager import get_audio_devices, get_default_devices


def test_audio_playback(device_id: int, duration: float = 2.0):
    """Test audio playback on specified device."""
    print(f"Testing audio playback on device {device_id}...")
    
    # Generate a simple test tone
    sample_rate = 44100
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a pleasant test tone (sine wave with fade in/out)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add fade in/out to prevent clicks
    fade_samples = int(0.1 * sample_rate)  # 100ms fade
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Scale to reasonable volume
    audio = (audio * 0.3).astype(np.float32)
    
    p = pyaudio.PyAudio()
    
    try:
        # Open stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=device_id,
            frames_per_buffer=1024
        )
        
        # Play audio in chunks
        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk with zeros
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            stream.write(chunk.tobytes())
        
        stream.stop_stream()
        stream.close()
        print("✓ Audio playback completed successfully")
        
    except Exception as e:
        print(f"✗ Audio playback failed: {e}")
    
    finally:
        p.terminate()


def test_audio_capture(device_id: int, duration: float = 3.0):
    """Test audio capture from specified device."""
    print(f"Testing audio capture from device {device_id} for {duration} seconds...")
    print("Please speak into your microphone...")
    
    sample_rate = 44100
    chunk_size = 1024
    
    p = pyaudio.PyAudio()
    
    try:
        # Open stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=chunk_size
        )
        
        print("Recording...")
        
        # Record audio
        frames = []
        num_chunks = int(sample_rate * duration / chunk_size)
        
        for i in range(num_chunks):
            data = stream.read(chunk_size)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)
            
            # Show simple level meter
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            level_bars = int(rms * 50)  # Scale for display
            level_display = "█" * level_bars + "░" * (20 - min(level_bars, 20))
            print(f"\rLevel: [{level_display}] {rms:.3f}", end="", flush=True)
        
        print("\n✓ Audio capture completed successfully")
        
        # Analyze captured audio
        all_audio = np.concatenate(frames)
        max_level = np.max(np.abs(all_audio))
        rms_level = np.sqrt(np.mean(all_audio ** 2))
        
        print(f"  Max level: {max_level:.3f}")
        print(f"  RMS level: {rms_level:.3f}")
        
        if max_level < 0.001:
            print("  ⚠️  Very low audio levels detected - check microphone connection")
        elif max_level > 0.9:
            print("  ⚠️  High audio levels detected - may be clipping")
        else:
            print("  ✓ Audio levels look good")
        
        stream.stop_stream()
        stream.close()
        
        return all_audio
        
    except Exception as e:
        print(f"✗ Audio capture failed: {e}")
        return None
    
    finally:
        p.terminate()


def test_capture_and_playback(input_device_id: int, output_device_id: int):
    """Test capturing audio and playing it back."""
    print(f"\nTesting capture from device {input_device_id} and playback to device {output_device_id}...")
    
    # Capture audio
    captured_audio = test_audio_capture(input_device_id, duration=3.0)
    
    if captured_audio is not None:
        print("Playing back captured audio...")
        time.sleep(1)  # Brief pause
        
        # Play back the captured audio
        sample_rate = 44100
        p = pyaudio.PyAudio()
        
        try:
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
            for i in range(0, len(captured_audio), chunk_size):
                chunk = captured_audio[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                stream.write(chunk.tobytes())
            
            stream.stop_stream()
            stream.close()
            print("✓ Playback of captured audio completed")
            
        except Exception as e:
            print(f"✗ Playback failed: {e}")
        
        finally:
            p.terminate()


def main():
    """Main test function."""
    print("=== Audio Device Test ===\n")
    
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
        
        # Get user selection
        if input_devices:
            while True:
                try:
                    choice = input(f"\nSelect input device (0-{len(input_devices)-1}): ").strip()
                    idx = int(choice)
                    if 0 <= idx < len(input_devices):
                        selected_input = input_devices[idx].id
                        break
                    else:
                        print(f"Please enter a number between 0 and {len(input_devices)-1}")
                except ValueError:
                    print("Please enter a valid number")
        
        if output_devices:
            while True:
                try:
                    choice = input(f"Select output device (0-{len(output_devices)-1}): ").strip()
                    idx = int(choice)
                    if 0 <= idx < len(output_devices):
                        selected_output = output_devices[idx].id
                        break
                    else:
                        print(f"Please enter a number between 0 and {len(output_devices)-1}")
                except ValueError:
                    print("Please enter a valid number")
        
        print(f"\nSelected input device: {selected_input}")
        print(f"Selected output device: {selected_output}")
        
        # Run tests
        print("\n" + "="*50)
        print("1. Testing audio playback...")
        test_audio_playback(selected_output, duration=2.0)
        
        print("\n" + "="*50)
        print("2. Testing audio capture...")
        test_audio_capture(selected_input, duration=3.0)
        
        print("\n" + "="*50)
        print("3. Testing capture and playback...")
        test_capture_and_playback(selected_input, selected_output)
        
        print("\n" + "="*50)
        print("Audio device test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()