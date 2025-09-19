#!/usr/bin/env python3
"""
Working Real-Time Voice Character Transformation Application

This is a functional voice transformation app that:
1. Records audio from your microphone
2. Converts speech to text using Whisper
3. Transforms text using character profiles
4. Synthesizes speech with character voice
5. Plays the transformed audio

This is a REAL working application, not a demo.
"""

import sys
import os
import asyncio
import threading
import time
import queue
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Core imports
import numpy as np
import pyaudio
import wave
import tempfile

# Try to import optional dependencies
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print("⚠️  Whisper not available - using mock STT")

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    print("⚠️  pyttsx3 not available - using mock TTS")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioRecorder:
    """Real-time audio recorder."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def start_recording(self):
        """Start recording audio."""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()
            logger.info("🎤 Audio recording started")
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        logger.info("🎤 Audio recording stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing."""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self, timeout=1.0):
        """Get next audio chunk."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def cleanup(self):
        """Cleanup audio resources."""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()


class SpeechToText:
    """Speech-to-text processor."""
    
    def __init__(self):
        self.model = None
        if HAS_WHISPER:
            try:
                logger.info("Loading Whisper model...")
                self.model = whisper.load_model("base")
                logger.info("✅ Whisper model loaded")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                self.model = None
    
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio data to text."""
        if self.model is None:
            # Mock transcription for testing
            return "This is mock transcribed text for testing purposes."
        
        try:
            # Convert audio data to format Whisper expects
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe
            result = self.model.transcribe(audio_float)
            text = result["text"].strip()
            
            if text:
                logger.info(f"🎯 Transcribed: {text}")
                return text
            else:
                return None
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None


class CharacterTransformer:
    """Character text transformation."""
    
    def __init__(self, characters_dir="characters"):
        self.characters_dir = Path(characters_dir)
        self.characters = self._load_characters()
        self.current_character = "default"
    
    def _load_characters(self):
        """Load character profiles."""
        characters = {}
        
        if not self.characters_dir.exists():
            logger.warning(f"Characters directory not found: {self.characters_dir}")
            return {"default": self._get_default_character()}
        
        for char_file in self.characters_dir.glob("*.json"):
            try:
                with open(char_file, 'r') as f:
                    char_data = json.load(f)
                characters[char_data["name"]] = char_data
                logger.info(f"📚 Loaded character: {char_data['name']}")
            except Exception as e:
                logger.error(f"Failed to load character {char_file}: {e}")
        
        if not characters:
            characters["default"] = self._get_default_character()
        
        return characters
    
    def _get_default_character(self):
        """Get default character profile."""
        return {
            "name": "default",
            "description": "Default character with no transformation",
            "personality_traits": ["neutral"],
            "speech_patterns": {},
            "vocabulary_preferences": {},
            "transformation_prompt": "Return the text as-is: {text}",
            "voice_model_path": "default"
        }
    
    def get_available_characters(self):
        """Get list of available characters."""
        return list(self.characters.keys())
    
    def set_character(self, character_name):
        """Set current character."""
        if character_name in self.characters:
            self.current_character = character_name
            logger.info(f"🎭 Character set to: {character_name}")
            return True
        else:
            logger.error(f"Character not found: {character_name}")
            return False
    
    def transform_text(self, text):
        """Transform text using current character."""
        if not text:
            return text
        
        character = self.characters[self.current_character]
        
        # Apply simple speech pattern transformations
        transformed = text
        for pattern, replacement in character.get("speech_patterns", {}).items():
            transformed = transformed.replace(pattern, replacement)
        
        # Add character-specific endings or modifications
        if character["name"] == "anime-waifu":
            if not transformed.endswith(("~", "!", "?")):
                transformed += " desu~"
        elif character["name"] == "patriotic-american":
            transformed = f"Well, {transformed}, fellow American!"
        elif character["name"] == "slurring-drunk":
            # Simple slurring effect
            transformed = transformed.replace("s", "sh").replace("the", "da")
        
        if transformed != text:
            logger.info(f"🎭 Transformed: {text} -> {transformed}")
        
        return transformed


class TextToSpeech:
    """Text-to-speech synthesizer."""
    
    def __init__(self):
        self.engine = None
        if HAS_TTS:
            try:
                self.engine = pyttsx3.init()
                # Set properties
                self.engine.setProperty('rate', 150)  # Speed
                self.engine.setProperty('volume', 0.8)  # Volume
                logger.info("✅ TTS engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                self.engine = None
    
    def speak_text(self, text, character="default"):
        """Convert text to speech and play it."""
        if not text:
            return
        
        if self.engine is None:
            logger.info(f"🔊 Mock TTS: {text}")
            return
        
        try:
            # Adjust voice properties based on character
            if character == "anime-waifu":
                self.engine.setProperty('rate', 180)  # Faster, more energetic
            elif character == "patriotic-american":
                self.engine.setProperty('rate', 140)  # Slower, more authoritative
            elif character == "slurring-drunk":
                self.engine.setProperty('rate', 120)  # Slower, slurred
            else:
                self.engine.setProperty('rate', 150)  # Default
            
            logger.info(f"🔊 Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            
        except Exception as e:
            logger.error(f"TTS failed: {e}")


class VoiceTransformationApp:
    """Main voice transformation application."""
    
    def __init__(self):
        self.recorder = AudioRecorder()
        self.stt = SpeechToText()
        self.character_transformer = CharacterTransformer()
        self.tts = TextToSpeech()
        
        self.is_running = False
        self.processing_thread = None
        
        # Audio processing parameters
        self.silence_threshold = 500  # Adjust based on your mic
        self.min_audio_length = 1.0  # Minimum seconds of audio to process
        self.audio_buffer = []
        self.last_speech_time = 0
        
    def start(self):
        """Start the voice transformation application."""
        logger.info("🚀 Starting Voice Character Transformation App")
        
        # Show available characters
        characters = self.character_transformer.get_available_characters()
        logger.info(f"📚 Available characters: {', '.join(characters)}")
        
        # Start audio recording
        if not self.recorder.start_recording():
            logger.error("❌ Failed to start audio recording")
            return False
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("✅ Voice transformation app started")
        logger.info("🎤 Speak into your microphone...")
        logger.info("⌨️  Commands: 'q' to quit, 'c' to change character")
        
        return True
    
    def stop(self):
        """Stop the voice transformation application."""
        logger.info("🛑 Stopping voice transformation app...")
        
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        self.recorder.cleanup()
        logger.info("✅ Voice transformation app stopped")
    
    def _processing_loop(self):
        """Main audio processing loop."""
        while self.is_running:
            try:
                # Get audio chunk
                audio_chunk = self.recorder.get_audio_chunk(timeout=0.1)
                if audio_chunk is None:
                    continue
                
                # Check for speech activity (simple volume-based detection)
                volume = np.sqrt(np.mean(audio_chunk**2))
                
                if volume > self.silence_threshold:
                    self.audio_buffer.append(audio_chunk)
                    self.last_speech_time = time.time()
                else:
                    # Check if we have enough audio to process
                    if (self.audio_buffer and 
                        time.time() - self.last_speech_time > 1.0 and
                        len(self.audio_buffer) * self.recorder.chunk_size / self.recorder.sample_rate > self.min_audio_length):
                        
                        # Process accumulated audio
                        self._process_audio_buffer()
                        self.audio_buffer = []
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
    
    def _process_audio_buffer(self):
        """Process accumulated audio buffer."""
        if not self.audio_buffer:
            return
        
        try:
            # Combine audio chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            logger.info("🎯 Processing speech...")
            
            # Speech to text
            text = self.stt.transcribe_audio(audio_data, self.recorder.sample_rate)
            if not text:
                return
            
            # Character transformation
            transformed_text = self.character_transformer.transform_text(text)
            
            # Text to speech
            self.tts.speak_text(transformed_text, self.character_transformer.current_character)
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
    
    def change_character(self):
        """Interactive character selection."""
        characters = self.character_transformer.get_available_characters()
        
        print("\n📚 Available Characters:")
        for i, char in enumerate(characters):
            current = " (current)" if char == self.character_transformer.current_character else ""
            print(f"  {i+1}. {char}{current}")
        
        try:
            choice = input("\nEnter character number: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(characters):
                    self.character_transformer.set_character(characters[idx])
                    print(f"✅ Character changed to: {characters[idx]}")
                else:
                    print("❌ Invalid character number")
            else:
                print("❌ Please enter a number")
        except (KeyboardInterrupt, EOFError):
            pass
    
    def run_interactive(self):
        """Run the app with interactive controls."""
        if not self.start():
            return
        
        try:
            while self.is_running:
                try:
                    command = input().strip().lower()
                    
                    if command == 'q' or command == 'quit':
                        break
                    elif command == 'c' or command == 'character':
                        self.change_character()
                    elif command == 'h' or command == 'help':
                        print("\n📋 Commands:")
                        print("  q, quit - Quit the application")
                        print("  c, character - Change character")
                        print("  h, help - Show this help")
                        print("  Just speak into your microphone for voice transformation!")
                    
                except (KeyboardInterrupt, EOFError):
                    break
                    
        finally:
            self.stop()


def install_dependencies():
    """Install missing dependencies."""
    missing = []
    
    if not HAS_WHISPER:
        missing.append("openai-whisper")
    if not HAS_TTS:
        missing.append("pyttsx3")
    
    if missing:
        print(f"📦 Installing missing dependencies: {', '.join(missing)}")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("✅ Dependencies installed successfully")
            print("🔄 Please restart the application")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True


def main():
    """Main entry point."""
    print("🎵 Real-Time Voice Character Transformation")
    print("=" * 50)
    
    # Check dependencies
    if not HAS_WHISPER or not HAS_TTS:
        print("⚠️  Missing optional dependencies for full functionality")
        print("   - Whisper (speech-to-text): pip install openai-whisper")
        print("   - pyttsx3 (text-to-speech): pip install pyttsx3")
        
        choice = input("\nInstall missing dependencies? (y/n): ").strip().lower()
        if choice == 'y':
            if install_dependencies():
                return 0
        
        print("\n🔄 Continuing with mock components...")
    
    # Create and run the app
    app = VoiceTransformationApp()
    
    print("\n🎮 Instructions:")
    print("1. Speak into your microphone")
    print("2. The app will transcribe your speech")
    print("3. Transform it using the selected character")
    print("4. Play back the transformed speech")
    print("\n📋 Commands while running:")
    print("  'c' - Change character")
    print("  'q' - Quit")
    print("  'h' - Help")
    
    input("\nPress Enter to start...")
    
    try:
        app.run_interactive()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())