#!/usr/bin/env python3
"""
Real Voice Character Transformation UI Application

This is a fully functional voice transformation app with GUI that:
1. Records audio from your microphone (with visual feedback)
2. Shows real-time transcription in the UI
3. Transforms text using character profiles
4. Plays back the transformed audio
5. Shows all processing steps in the interface

This is a REAL working application with UI, not a demo.
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
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pyaudio
import wave
import tempfile

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Try to import optional dependencies
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Real-time audio processing with visual feedback."""
    
    def __init__(self, ui_callback=None):
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.ui_callback = ui_callback
        
        # Voice activity detection
        self.silence_threshold = 500
        self.min_audio_length = 1.0
        self.audio_buffer = []
        self.last_speech_time = 0
        self.current_volume = 0
        
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
            if self.ui_callback:
                self.ui_callback("recording_started", True)
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            if self.ui_callback:
                self.ui_callback("error", f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        logger.info("🎤 Audio recording stopped")
        if self.ui_callback:
            self.ui_callback("recording_started", False)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing."""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Calculate volume for visual feedback
            try:
                volume = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                self.current_volume = min(100, volume / 10)  # Scale to 0-100
            except:
                self.current_volume = 0
            
            # Send volume to UI
            if self.ui_callback:
                self.ui_callback("audio_level", self.current_volume)
            
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self, timeout=0.1):
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


class VoiceTransformationEngine:
    """Core voice transformation engine."""
    
    def __init__(self, ui_callback=None):
        self.ui_callback = ui_callback
        self.stt_model = None
        self.tts_engine = None
        self.characters = {}
        self.current_character = "default"
        
        # Initialize components
        self._init_stt()
        self._init_tts()
        self._load_characters()
    
    def _init_stt(self):
        """Initialize speech-to-text."""
        if HAS_WHISPER:
            try:
                if self.ui_callback:
                    self.ui_callback("status", "Loading Whisper model...")
                logger.info("Loading Whisper model...")
                self.stt_model = whisper.load_model("base")
                logger.info("✅ Whisper model loaded")
                if self.ui_callback:
                    self.ui_callback("status", "Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                if self.ui_callback:
                    self.ui_callback("error", f"Failed to load Whisper: {e}")
        else:
            if self.ui_callback:
                self.ui_callback("status", "Using mock STT (Whisper not available)")
    
    def _init_tts(self):
        """Initialize text-to-speech."""
        if HAS_TTS:
            try:
                logger.info("Initializing TTS engine...")
                
                # Initialize with minimal settings
                self.tts_engine = pyttsx3.init()
                
                # Set basic properties only
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                
                # Get voice info for debugging
                try:
                    voices = self.tts_engine.getProperty('voices')
                    if voices:
                        logger.info(f"Available voices: {len(voices)}")
                        current_voice = self.tts_engine.getProperty('voice')
                        logger.info(f"Current voice: {current_voice}")
                    else:
                        logger.warning("No voices found")
                except Exception as voice_error:
                    logger.warning(f"Could not get voice info: {voice_error}")
                
                # Simple test without blocking initialization
                logger.info("✅ TTS engine initialized")
                if self.ui_callback:
                    self.ui_callback("status", "TTS engine initialized")
                
                # Test in background
                def test_tts():
                    try:
                        time.sleep(0.5)  # Small delay
                        self.tts_engine.say("TTS engine ready")
                        self.tts_engine.runAndWait()
                        logger.info("✅ TTS test completed")
                    except Exception as test_error:
                        logger.warning(f"TTS test failed: {test_error}")
                
                threading.Thread(target=test_tts, daemon=True).start()
                    
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                import traceback
                traceback.print_exc()
                self.tts_engine = None
                if self.ui_callback:
                    self.ui_callback("error", f"Failed to initialize TTS: {e}")
        else:
            logger.warning("pyttsx3 not available, using mock TTS")
            if self.ui_callback:
                self.ui_callback("status", "Using mock TTS (pyttsx3 not available)")
    
    def _load_characters(self):
        """Load character profiles."""
        characters_dir = Path("characters")
        
        if not characters_dir.exists():
            self.characters = {"default": self._get_default_character()}
            return
        
        for char_file in characters_dir.glob("*.json"):
            try:
                with open(char_file, 'r') as f:
                    char_data = json.load(f)
                self.characters[char_data["name"]] = char_data
                logger.info(f"📚 Loaded character: {char_data['name']}")
            except Exception as e:
                logger.error(f"Failed to load character {char_file}: {e}")
        
        if not self.characters:
            self.characters["default"] = self._get_default_character()
        
        if self.ui_callback:
            self.ui_callback("characters_loaded", list(self.characters.keys()))
    
    def _get_default_character(self):
        """Get default character profile."""
        return {
            "name": "default",
            "description": "Default character with no transformation",
            "personality_traits": ["neutral"],
            "speech_patterns": {},
            "transformation_prompt": "Return the text as-is: {text}",
        }
    
    def get_available_characters(self):
        """Get list of available characters."""
        return list(self.characters.keys())
    
    def set_character(self, character_name):
        """Set current character."""
        if character_name in self.characters:
            self.current_character = character_name
            logger.info(f"🎭 Character set to: {character_name}")
            if self.ui_callback:
                self.ui_callback("character_changed", character_name)
            return True
        return False
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio to text."""
        try:
            if self.ui_callback:
                self.ui_callback("status", "Transcribing speech...")
            
            # Convert audio data to format Whisper expects
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Audio quality analysis
            max_amplitude = np.max(np.abs(audio_float))
            rms_amplitude = np.sqrt(np.mean(audio_float**2))
            duration = len(audio_float) / 16000
            
            logger.info(f"Audio analysis: duration={duration:.2f}s, max_amp={max_amplitude:.4f}, rms_amp={rms_amplitude:.4f}")
            
            if self.ui_callback:
                self.ui_callback("status", f"Audio: {duration:.1f}s, amplitude: {max_amplitude:.3f}")
            
            if self.stt_model is None:
                # Mock transcription for testing - but make it realistic
                if max_amplitude > 0.01 and duration > 1.0:
                    mock_text = "This is a mock transcription since Whisper is not available"
                    logger.info(f"🎯 Mock Transcribed: {mock_text}")
                    if self.ui_callback:
                        self.ui_callback("transcription", mock_text)
                    return mock_text
                else:
                    logger.info("Mock: Audio too quiet or short")
                    if self.ui_callback:
                        self.ui_callback("status", "Audio too quiet or short for transcription")
                    return None
            
            # Check if audio is suitable for transcription
            if max_amplitude < 0.005:
                logger.warning(f"Audio too quiet: max_amplitude={max_amplitude:.6f}")
                if self.ui_callback:
                    self.ui_callback("status", "Audio too quiet - speak louder")
                return None
            
            if duration < 0.5:
                logger.warning(f"Audio too short: duration={duration:.2f}s")
                if self.ui_callback:
                    self.ui_callback("status", "Audio too short - speak longer")
                return None
            
            # Normalize audio to improve transcription
            if max_amplitude > 0:
                audio_float = audio_float / max_amplitude * 0.8  # Normalize to 80% of max
            
            logger.info(f"Sending to Whisper: {len(audio_float)} samples, normalized")
            
            # Transcribe with Whisper - more aggressive settings
            result = self.stt_model.transcribe(
                audio_float,
                language="en",
                task="transcribe",
                verbose=True,  # Enable verbose for debugging
                temperature=0.0,  # More deterministic
                best_of=1,  # Faster processing
                beam_size=1,  # Faster processing
                word_timestamps=False,
                fp16=False  # Use fp32 for better accuracy
            )
            
            text = result["text"].strip()
            confidence = result.get("confidence", 0.0)
            
            logger.info(f"Whisper result: '{text}' (confidence: {confidence})")
            
            # More lenient text filtering
            if text and len(text.strip()) > 0:
                # Filter out common Whisper artifacts
                if text.lower() in ["you", "thank you", "thanks", "bye", ".", "?", "!"]:
                    logger.info(f"Filtered out common artifact: '{text}'")
                    if self.ui_callback:
                        self.ui_callback("status", "Filtered out transcription artifact")
                    return None
                
                logger.info(f"🎯 Successfully transcribed: '{text}'")
                if self.ui_callback:
                    self.ui_callback("transcription", text)
                return text
            else:
                logger.info("Whisper returned empty text")
                if self.ui_callback:
                    self.ui_callback("status", "No speech detected by Whisper")
                return None
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            if self.ui_callback:
                self.ui_callback("error", f"Transcription failed: {e}")
            return None
    
    def transform_text(self, text):
        """Transform text using current character."""
        if not text:
            return text
        
        try:
            if self.ui_callback:
                self.ui_callback("status", f"Transforming text as {self.current_character}...")
            
            character = self.characters[self.current_character]
            
            # Apply simple speech pattern transformations
            transformed = text
            for pattern, replacement in character.get("speech_patterns", {}).items():
                transformed = transformed.replace(pattern, replacement)
            
            # Add character-specific modifications
            if character["name"] == "anime-waifu":
                if not transformed.endswith(("~", "!", "?")):
                    transformed += " desu~"
            elif character["name"] == "patriotic-american":
                transformed = f"Well, {transformed}, fellow American!"
            elif character["name"] == "slurring-drunk":
                transformed = transformed.replace("s", "sh").replace("the", "da")
            
            if transformed != text:
                logger.info(f"🎭 Transformed: {text} -> {transformed}")
                if self.ui_callback:
                    self.ui_callback("transformation", transformed)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Text transformation failed: {e}")
            if self.ui_callback:
                self.ui_callback("error", f"Text transformation failed: {e}")
            return text
    
    def speak_text(self, text):
        """Convert text to speech and play it."""
        if not text:
            return
        
        try:
            if self.ui_callback:
                self.ui_callback("status", "Generating speech...")
            
            logger.info(f"🔊 TTS Request: '{text}'")
            
            if self.tts_engine is None:
                logger.info(f"🔊 Mock TTS (no engine): {text}")
                if self.ui_callback:
                    self.ui_callback("status", f"Mock TTS: {text}")
                # Simulate TTS delay
                time.sleep(len(text) * 0.05)  # ~50ms per character
                if self.ui_callback:
                    self.ui_callback("status", "Mock speech completed")
                return
            
            # Simple, direct approach - don't change settings during playback
            logger.info(f"🔊 Starting TTS playback: '{text}'")
            if self.ui_callback:
                self.ui_callback("status", f"🔊 Speaking: {text}")
            
            # Use the simplest possible approach
            try:
                # Just speak the text directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                
                logger.info("🔊 TTS playback completed successfully")
                if self.ui_callback:
                    self.ui_callback("status", "✅ Speech playback completed")
                    
            except Exception as tts_error:
                logger.error(f"TTS playback error: {tts_error}")
                import traceback
                traceback.print_exc()
                
                # Try with a fresh engine instance
                try:
                    logger.info("🔄 Trying with fresh TTS engine...")
                    fresh_engine = pyttsx3.init()
                    fresh_engine.setProperty('rate', 150)
                    fresh_engine.setProperty('volume', 0.9)
                    fresh_engine.say(text)
                    fresh_engine.runAndWait()
                    
                    logger.info("🔊 TTS playback completed (with fresh engine)")
                    if self.ui_callback:
                        self.ui_callback("status", "✅ Speech completed (recovered)")
                        
                except Exception as retry_error:
                    logger.error(f"Fresh TTS engine also failed: {retry_error}")
                    import traceback
                    traceback.print_exc()
                    if self.ui_callback:
                        self.ui_callback("error", f"TTS failed completely: {retry_error}")
            
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            import traceback
            traceback.print_exc()
            if self.ui_callback:
                self.ui_callback("error", f"TTS failed: {e}")


class VoiceTransformationUI:
    """Main UI for voice transformation application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-Time Voice Character Transformation")
        self.root.geometry("800x600")
        
        # Components
        self.audio_processor = None
        self.voice_engine = None
        self.processing_thread = None
        self.is_running = False
        
        # UI Variables
        self.current_character = tk.StringVar(value="default")
        self.audio_level = tk.DoubleVar()
        self.is_recording = tk.BooleanVar()
        
        self._create_ui()
        self._init_components()
    
    def _create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎵 Real-Time Voice Character Transformation", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Character selection
        ttk.Label(main_frame, text="Character:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.character_combo = ttk.Combobox(main_frame, textvariable=self.current_character, 
                                          state="readonly", width=20)
        self.character_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.character_combo.bind('<<ComboboxSelected>>', self._on_character_changed)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=2, sticky=tk.E)
        
        self.start_button = ttk.Button(button_frame, text="🎤 Start Recording", 
                                     command=self._toggle_recording)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Test button for debugging
        self.test_button = ttk.Button(button_frame, text="🧪 Test TTS", 
                                    command=self._test_tts)
        self.test_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Audio level indicator
        ttk.Label(main_frame, text="Audio Level:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.audio_level_bar = ttk.Progressbar(main_frame, variable=self.audio_level, 
                                             maximum=100, length=200)
        self.audio_level_bar.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(10, 0), padx=(0, 10))
        
        # Recording indicator
        self.recording_label = ttk.Label(main_frame, text="⚫ Not Recording", 
                                       foreground="gray")
        self.recording_label.grid(row=2, column=2, sticky=tk.E, pady=(10, 0))
        
        # Status
        ttk.Label(main_frame, text="Status:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # Text displays
        text_frame = ttk.LabelFrame(main_frame, text="Processing Results", padding="10")
        text_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)
        text_frame.rowconfigure(3, weight=1)
        
        # Original text
        ttk.Label(text_frame, text="🎯 Original Speech (Transcription):").grid(row=0, column=0, sticky=tk.W)
        self.original_text = scrolledtext.ScrolledText(text_frame, height=6, wrap=tk.WORD)
        self.original_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 10))
        
        # Transformed text
        ttk.Label(text_frame, text="🎭 Transformed Speech:").grid(row=2, column=0, sticky=tk.W)
        self.transformed_text = scrolledtext.ScrolledText(text_frame, height=6, wrap=tk.WORD)
        self.transformed_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        
        # Configure text frame grid weights
        main_frame.rowconfigure(4, weight=1)
    
    def _init_components(self):
        """Initialize voice processing components."""
        try:
            self.voice_engine = VoiceTransformationEngine(ui_callback=self._handle_engine_callback)
            self.audio_processor = AudioProcessor(ui_callback=self._handle_audio_callback)
            
            # Update character list
            characters = self.voice_engine.get_available_characters()
            self.character_combo['values'] = characters
            if characters:
                self.current_character.set(characters[0])
                self.voice_engine.set_character(characters[0])
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize components: {e}")
    
    def _handle_engine_callback(self, event_type, data):
        """Handle callbacks from the voice engine."""
        if event_type == "status":
            self.root.after(0, lambda: self.status_label.config(text=data, foreground="blue"))
        elif event_type == "error":
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {data}", foreground="red"))
        elif event_type == "transcription":
            self.root.after(0, lambda: self._append_text(self.original_text, f"🎯 {data}\n"))
        elif event_type == "transformation":
            self.root.after(0, lambda: self._append_text(self.transformed_text, f"🎭 {data}\n"))
        elif event_type == "characters_loaded":
            self.root.after(0, lambda: self.character_combo.config(values=data))
    
    def _handle_audio_callback(self, event_type, data):
        """Handle callbacks from the audio processor."""
        if event_type == "audio_level":
            self.root.after(0, lambda: self.audio_level.set(data))
        elif event_type == "recording_started":
            if data:
                self.root.after(0, lambda: self.recording_label.config(text="🔴 Recording", foreground="red"))
            else:
                self.root.after(0, lambda: self.recording_label.config(text="⚫ Not Recording", foreground="gray"))
        elif event_type == "error":
            self.root.after(0, lambda: messagebox.showerror("Audio Error", data))
    
    def _append_text(self, text_widget, text):
        """Append text to a text widget."""
        text_widget.insert(tk.END, text)
        text_widget.see(tk.END)
    
    def _on_character_changed(self, event=None):
        """Handle character selection change."""
        character = self.current_character.get()
        if self.voice_engine:
            self.voice_engine.set_character(character)
            self.status_label.config(text=f"Character changed to: {character}", foreground="green")
    
    def _toggle_recording(self):
        """Toggle recording on/off."""
        if not self.is_running:
            self._start_processing()
        else:
            self._stop_processing()
    
    def _start_processing(self):
        """Start voice processing."""
        try:
            if not self.audio_processor.start_recording():
                return
            
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.start_button.config(text="🛑 Stop Recording")
            self.status_label.config(text="Recording started - speak into your microphone!", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Start Error", f"Failed to start processing: {e}")
    
    def _stop_processing(self):
        """Stop voice processing."""
        self.is_running = False
        
        if self.audio_processor:
            self.audio_processor.stop_recording()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        self.start_button.config(text="🎤 Start Recording")
        self.status_label.config(text="Recording stopped", foreground="gray")
    
    def _processing_loop(self):
        """Main audio processing loop."""
        audio_buffer = []
        last_speech_time = 0
        silence_threshold = 50   # Even lower threshold
        min_audio_length = 1.0   # Shorter minimum length
        max_buffer_time = 8.0    # Shorter max time
        silence_duration = 1.5   # Shorter silence duration
        buffer_start_time = None
        volume_history = []
        
        while self.is_running:
            try:
                # Get audio chunk
                audio_chunk = self.audio_processor.get_audio_chunk(timeout=0.1)
                if audio_chunk is None:
                    continue
                
                # Calculate volume with better method
                try:
                    # Use RMS for better volume detection
                    volume = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
                    volume_history.append(volume)
                    if len(volume_history) > 100:  # Keep last 100 samples
                        volume_history.pop(0)
                    
                    # Adaptive threshold based on background noise
                    if len(volume_history) > 50:
                        avg_volume = np.mean(volume_history)
                        adaptive_threshold = max(silence_threshold, avg_volume * 2)
                    else:
                        adaptive_threshold = silence_threshold
                        
                except:
                    volume = 0
                    adaptive_threshold = silence_threshold
                
                # Debug: Show volume levels more frequently
                if len(audio_buffer) % 20 == 0:  # Every ~2 seconds
                    self.root.after(0, lambda v=volume, t=adaptive_threshold: self.status_label.config(
                        text=f"Listening... (vol: {v:.1f}, thresh: {t:.1f})", 
                        foreground="blue"))
                
                if volume > adaptive_threshold:
                    if not audio_buffer:  # Starting new recording
                        buffer_start_time = time.time()
                        self.root.after(0, lambda: self.status_label.config(
                            text="🎤 Speech detected - recording...", foreground="orange"))
                    
                    audio_buffer.append(audio_chunk)
                    last_speech_time = time.time()
                else:
                    # Check if we should process the buffer
                    current_time = time.time()
                    buffer_duration = len(audio_buffer) * self.audio_processor.chunk_size / self.audio_processor.sample_rate
                    
                    should_process = False
                    reason = ""
                    
                    if audio_buffer:
                        # Process if we have silence after speech
                        if current_time - last_speech_time > silence_duration and buffer_duration >= min_audio_length:
                            should_process = True
                            reason = f"silence after {buffer_duration:.1f}s of speech"
                        # Process if buffer is getting too long
                        elif buffer_start_time and current_time - buffer_start_time > max_buffer_time:
                            should_process = True
                            reason = f"max buffer time ({buffer_duration:.1f}s)"
                        # Process if we have a good amount of audio
                        elif buffer_duration >= 3.0:  # Process longer audio immediately
                            should_process = True
                            reason = f"sufficient audio ({buffer_duration:.1f}s)"
                    
                    if should_process:
                        self.root.after(0, lambda r=reason: self.status_label.config(
                            text=f"🎯 Processing: {r}...", foreground="green"))
                        
                        # Process in a separate thread to avoid blocking
                        threading.Thread(target=self._process_audio_buffer, 
                                       args=(audio_buffer.copy(),), daemon=True).start()
                        audio_buffer = []
                        buffer_start_time = None
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.root.after(0, lambda e=str(e): self.status_label.config(
                    text=f"Processing error: {e}", foreground="red"))
                time.sleep(0.1)
    
    def _process_audio_buffer(self, audio_buffer):
        """Process accumulated audio buffer."""
        if not audio_buffer:
            return
        
        try:
            # Combine audio chunks
            audio_data = np.concatenate(audio_buffer)
            buffer_duration = len(audio_data) / self.audio_processor.sample_rate
            
            self.root.after(0, lambda: self.status_label.config(
                text=f"🎯 Processing {buffer_duration:.1f}s of audio...", foreground="blue"))
            
            logger.info(f"Processing audio buffer: {len(audio_data)} samples, {buffer_duration:.1f}s")
            
            # Speech to text
            text = self.voice_engine.transcribe_audio(audio_data)
            if not text or text.strip() == "":
                self.root.after(0, lambda: self.status_label.config(
                    text="No speech detected in audio", foreground="orange"))
                return
            
            logger.info(f"Transcription successful: {text}")
            
            # Character transformation
            transformed_text = self.voice_engine.transform_text(text)
            
            # Text to speech (run in background to avoid blocking)
            threading.Thread(target=self.voice_engine.speak_text, 
                           args=(transformed_text,), daemon=True).start()
            
            self.root.after(0, lambda: self.status_label.config(
                text="✅ Processing complete", foreground="green"))
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda e=str(e): self.status_label.config(
                text=f"Processing error: {e}", foreground="red"))
    
    def run(self):
        """Run the application."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self._on_closing()
    
    def _test_tts(self):
        """Test TTS with sample text."""
        def run_tts_test():
            try:
                self.root.after(0, lambda: self.status_label.config(text="Testing TTS...", foreground="blue"))
                
                # Test with simple text first
                simple_text = "Hello world"
                logger.info(f"Testing TTS with: '{simple_text}'")
                
                # Add to UI
                self.root.after(0, lambda: self._append_text(self.original_text, f"🧪 SIMPLE TEST: {simple_text}\n"))
                
                if self.voice_engine and self.voice_engine.tts_engine:
                    # Direct TTS test
                    logger.info("Direct TTS test starting...")
                    self.voice_engine.tts_engine.say(simple_text)
                    self.voice_engine.tts_engine.runAndWait()
                    logger.info("Direct TTS test completed")
                    
                    # Now test with character transformation
                    test_text = f"This is a test of the {self.current_character.get()} character."
                    self.root.after(0, lambda: self._append_text(self.transformed_text, f"🧪 CHARACTER TEST: {test_text}\n"))
                    
                    # Transform and speak
                    transformed = self.voice_engine.transform_text(test_text)
                    self.voice_engine.speak_text(transformed)
                    
                    self.root.after(0, lambda: self.status_label.config(text="✅ TTS test completed", foreground="green"))
                else:
                    self.root.after(0, lambda: self.status_label.config(text="❌ No TTS engine available", foreground="red"))
                    
            except Exception as e:
                logger.error(f"TTS test failed: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.status_label.config(text=f"TTS test failed: {e}", foreground="red"))
        
        # Run in background thread
        threading.Thread(target=run_tts_test, daemon=True).start()
    
    def _on_closing(self):
        """Handle application closing."""
        self._stop_processing()
        if self.audio_processor:
            self.audio_processor.cleanup()
        self.root.destroy()


def main():
    """Main entry point."""
    print("🎵 Real-Time Voice Character Transformation UI")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = []
    if not HAS_WHISPER:
        missing_deps.append("openai-whisper")
    if not HAS_TTS:
        missing_deps.append("pyttsx3")
    
    if missing_deps:
        print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("The app will work with mock components for missing dependencies.")
        print("Install with: pip install " + " ".join(missing_deps))
    
    try:
        app = VoiceTransformationUI()
        app.run()
    except Exception as e:
        print(f"❌ Application failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())