"""
Settings dialog for the Voice Character Transformation application.

This module provides a settings dialog window for configuring audio devices,
character transformation parameters, and other application settings.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable
import logging

try:
    # Try relative imports first (when used as a package)
    from ..audio.device_manager import AudioDeviceManager, get_audio_devices
    from ..core.config import ConfigManager, AppConfig, AudioConfig, CharacterConfig, STTConfig, TTSConfig, PerformanceConfig, LoggingConfig
    from ..core.interfaces import AudioDevice
except ImportError:
    # Fallback to absolute imports (when used as a script)
    from audio.device_manager import AudioDeviceManager, get_audio_devices
    from core.config import ConfigManager, AppConfig, AudioConfig, CharacterConfig, STTConfig, TTSConfig, PerformanceConfig, LoggingConfig
    from core.interfaces import AudioDevice

logger = logging.getLogger(__name__)


class SettingsDialog:
    """Settings dialog for application configuration."""
    
    def __init__(self, parent: tk.Tk, config_manager: ConfigManager, current_config: AppConfig):
        """
        Initialize the settings dialog.
        
        Args:
            parent: Parent window
            config_manager: Configuration manager instance
            current_config: Current application configuration
        """
        self.parent = parent
        self.config_manager = config_manager
        self.current_config = current_config
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("600x500")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog on parent
        self._center_dialog()
        
        # Configuration variables
        self._setup_variables()
        
        # Audio device lists
        self.input_devices = []
        self.output_devices = []
        
        # Callbacks
        self.on_settings_changed: Optional[Callable[[AppConfig], None]] = None
        
        # Create UI
        self._setup_ui()
        self._load_current_settings()
        self._load_audio_devices()
        
        # Handle dialog close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
    def _center_dialog(self):
        """Center the dialog on the parent window."""
        self.dialog.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_reqwidth()
        dialog_height = self.dialog.winfo_reqheight()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def _setup_variables(self):
        """Set up Tkinter variables for settings."""
        # Audio settings
        self.input_device_var = tk.StringVar()
        self.output_device_var = tk.StringVar()
        self.sample_rate_var = tk.IntVar()
        self.chunk_size_var = tk.IntVar()
        self.vad_threshold_var = tk.DoubleVar()
        
        # Character settings
        self.intensity_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        self.temperature_var = tk.DoubleVar()
        
        # STT settings
        self.stt_model_size_var = tk.StringVar()
        self.stt_device_var = tk.StringVar()
        self.stt_language_var = tk.StringVar()
        
        # TTS settings
        self.tts_device_var = tk.StringVar()
        self.tts_speed_var = tk.DoubleVar()
        
        # Performance settings
        self.max_latency_var = tk.IntVar()
        self.gpu_memory_var = tk.DoubleVar()
        self.enable_offloading_var = tk.BooleanVar()
        
        # Logging settings
        self.log_level_var = tk.StringVar()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_audio_tab(notebook)
        self._create_character_tab(notebook)
        self._create_stt_tab(notebook)
        self._create_tts_tab(notebook)
        self._create_performance_tab(notebook)
        self._create_logging_tab(notebook)
        
        # Create button frame
        self._create_button_frame()
    
    def _create_audio_tab(self, notebook: ttk.Notebook):
        """Create audio settings tab."""
        audio_frame = ttk.Frame(notebook)
        notebook.add(audio_frame, text="Audio")
        
        # Main frame with scrollbar
        canvas = tk.Canvas(audio_frame)
        scrollbar = ttk.Scrollbar(audio_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Audio device selection
        device_frame = ttk.LabelFrame(scrollable_frame, text="Audio Devices", padding="10")
        device_frame.pack(fill=tk.X, pady=(0, 10))
        device_frame.columnconfigure(1, weight=1)
        
        # Input device
        ttk.Label(device_frame, text="Input Device:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.input_device_combo = ttk.Combobox(
            device_frame,
            textvariable=self.input_device_var,
            state="readonly",
            width=40
        )
        self.input_device_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(
            device_frame,
            text="Refresh",
            command=self._refresh_audio_devices,
            width=10
        ).grid(row=0, column=2)
        
        # Output device
        ttk.Label(device_frame, text="Output Device:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.output_device_combo = ttk.Combobox(
            device_frame,
            textvariable=self.output_device_var,
            state="readonly",
            width=40
        )
        self.output_device_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(5, 0))
        
        # Audio parameters
        params_frame = ttk.LabelFrame(scrollable_frame, text="Audio Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        
        # Sample rate
        ttk.Label(params_frame, text="Sample Rate:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        sample_rate_combo = ttk.Combobox(
            params_frame,
            textvariable=self.sample_rate_var,
            values=[8000, 16000, 22050, 44100, 48000],
            state="readonly",
            width=15
        )
        sample_rate_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Chunk size
        ttk.Label(params_frame, text="Chunk Size:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        chunk_size_spin = ttk.Spinbox(
            params_frame,
            from_=256,
            to=8192,
            increment=256,
            textvariable=self.chunk_size_var,
            width=15
        )
        chunk_size_spin.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # VAD threshold
        ttk.Label(params_frame, text="VAD Threshold:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        vad_frame = ttk.Frame(params_frame)
        vad_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        vad_frame.columnconfigure(0, weight=1)
        
        vad_scale = ttk.Scale(
            vad_frame,
            from_=0.0,
            to=1.0,
            variable=self.vad_threshold_var,
            orient=tk.HORIZONTAL
        )
        vad_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.vad_label = ttk.Label(vad_frame, text="0.5")
        self.vad_label.grid(row=0, column=1)
        
        # Update label when scale changes
        vad_scale.configure(command=self._update_vad_label)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_character_tab(self, notebook: ttk.Notebook):
        """Create character transformation settings tab."""
        char_frame = ttk.Frame(notebook)
        notebook.add(char_frame, text="Character")
        
        # Character transformation frame
        transform_frame = ttk.LabelFrame(char_frame, text="Character Transformation", padding="10")
        transform_frame.pack(fill=tk.X, padx=10, pady=10)
        transform_frame.columnconfigure(1, weight=1)
        
        # Intensity slider
        ttk.Label(transform_frame, text="Transformation Intensity:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        intensity_frame = ttk.Frame(transform_frame)
        intensity_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        intensity_frame.columnconfigure(0, weight=1)
        
        intensity_scale = ttk.Scale(
            intensity_frame,
            from_=0.0,
            to=2.0,
            variable=self.intensity_var,
            orient=tk.HORIZONTAL
        )
        intensity_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.intensity_label = ttk.Label(intensity_frame, text="1.0")
        self.intensity_label.grid(row=0, column=1)
        
        # Update label when scale changes
        intensity_scale.configure(command=self._update_intensity_label)
        
        # LLM parameters frame
        llm_frame = ttk.LabelFrame(char_frame, text="LLM Parameters", padding="10")
        llm_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        llm_frame.columnconfigure(1, weight=1)
        
        # Max tokens
        ttk.Label(llm_frame, text="Max Tokens:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        max_tokens_spin = ttk.Spinbox(
            llm_frame,
            from_=1,
            to=4096,
            increment=32,
            textvariable=self.max_tokens_var,
            width=15
        )
        max_tokens_spin.grid(row=0, column=1, sticky=tk.W)
        
        # Temperature
        ttk.Label(llm_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        temp_frame = ttk.Frame(llm_frame)
        temp_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        temp_frame.columnconfigure(0, weight=1)
        
        temp_scale = ttk.Scale(
            temp_frame,
            from_=0.0,
            to=2.0,
            variable=self.temperature_var,
            orient=tk.HORIZONTAL
        )
        temp_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.temperature_label = ttk.Label(temp_frame, text="0.7")
        self.temperature_label.grid(row=0, column=1)
        
        # Update label when scale changes
        temp_scale.configure(command=self._update_temperature_label)
    
    def _create_stt_tab(self, notebook: ttk.Notebook):
        """Create Speech-to-Text settings tab."""
        stt_frame = ttk.Frame(notebook)
        notebook.add(stt_frame, text="Speech-to-Text")
        
        # STT parameters frame
        stt_params_frame = ttk.LabelFrame(stt_frame, text="STT Parameters", padding="10")
        stt_params_frame.pack(fill=tk.X, padx=10, pady=10)
        stt_params_frame.columnconfigure(1, weight=1)
        
        # Model size
        ttk.Label(stt_params_frame, text="Model Size:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        model_size_combo = ttk.Combobox(
            stt_params_frame,
            textvariable=self.stt_model_size_var,
            values=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
            state="readonly",
            width=15
        )
        model_size_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Device
        ttk.Label(stt_params_frame, text="Device:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        device_combo = ttk.Combobox(
            stt_params_frame,
            textvariable=self.stt_device_var,
            values=["cuda", "cpu"],
            state="readonly",
            width=15
        )
        device_combo.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Language
        ttk.Label(stt_params_frame, text="Language:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        language_entry = ttk.Entry(
            stt_params_frame,
            textvariable=self.stt_language_var,
            width=15
        )
        language_entry.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Language help
        ttk.Label(
            stt_params_frame,
            text="(Use 'auto' for automatic detection or language code like 'en', 'es', etc.)",
            font=("TkDefaultFont", 8),
            foreground="gray"
        ).grid(row=3, column=1, sticky=tk.W, pady=(2, 0))
    
    def _create_tts_tab(self, notebook: ttk.Notebook):
        """Create Text-to-Speech settings tab."""
        tts_frame = ttk.Frame(notebook)
        notebook.add(tts_frame, text="Text-to-Speech")
        
        # TTS parameters frame
        tts_params_frame = ttk.LabelFrame(tts_frame, text="TTS Parameters", padding="10")
        tts_params_frame.pack(fill=tk.X, padx=10, pady=10)
        tts_params_frame.columnconfigure(1, weight=1)
        
        # Device
        ttk.Label(tts_params_frame, text="Device:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        tts_device_combo = ttk.Combobox(
            tts_params_frame,
            textvariable=self.tts_device_var,
            values=["cuda", "cpu"],
            state="readonly",
            width=15
        )
        tts_device_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Speed
        ttk.Label(tts_params_frame, text="Speech Speed:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        speed_frame = ttk.Frame(tts_params_frame)
        speed_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        speed_frame.columnconfigure(0, weight=1)
        
        speed_scale = ttk.Scale(
            speed_frame,
            from_=0.1,
            to=3.0,
            variable=self.tts_speed_var,
            orient=tk.HORIZONTAL
        )
        speed_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.speed_label = ttk.Label(speed_frame, text="1.0")
        self.speed_label.grid(row=0, column=1)
        
        # Update label when scale changes
        speed_scale.configure(command=self._update_speed_label)
    
    def _create_performance_tab(self, notebook: ttk.Notebook):
        """Create performance settings tab."""
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance")
        
        # Performance parameters frame
        perf_params_frame = ttk.LabelFrame(perf_frame, text="Performance Parameters", padding="10")
        perf_params_frame.pack(fill=tk.X, padx=10, pady=10)
        perf_params_frame.columnconfigure(1, weight=1)
        
        # Max latency
        ttk.Label(perf_params_frame, text="Max Latency (ms):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        latency_spin = ttk.Spinbox(
            perf_params_frame,
            from_=100,
            to=10000,
            increment=100,
            textvariable=self.max_latency_var,
            width=15
        )
        latency_spin.grid(row=0, column=1, sticky=tk.W)
        
        # GPU memory fraction
        ttk.Label(perf_params_frame, text="GPU Memory Fraction:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        gpu_frame = ttk.Frame(perf_params_frame)
        gpu_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        gpu_frame.columnconfigure(0, weight=1)
        
        gpu_scale = ttk.Scale(
            gpu_frame,
            from_=0.1,
            to=1.0,
            variable=self.gpu_memory_var,
            orient=tk.HORIZONTAL
        )
        gpu_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.gpu_label = ttk.Label(gpu_frame, text="0.8")
        self.gpu_label.grid(row=0, column=1)
        
        # Update label when scale changes
        gpu_scale.configure(command=self._update_gpu_label)
        
        # Model offloading
        ttk.Checkbutton(
            perf_params_frame,
            text="Enable Model Offloading",
            variable=self.enable_offloading_var
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
    
    def _create_logging_tab(self, notebook: ttk.Notebook):
        """Create logging settings tab."""
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Logging")
        
        # Logging parameters frame
        log_params_frame = ttk.LabelFrame(log_frame, text="Logging Parameters", padding="10")
        log_params_frame.pack(fill=tk.X, padx=10, pady=10)
        log_params_frame.columnconfigure(1, weight=1)
        
        # Log level
        ttk.Label(log_params_frame, text="Log Level:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        log_level_combo = ttk.Combobox(
            log_params_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            state="readonly",
            width=15
        )
        log_level_combo.grid(row=0, column=1, sticky=tk.W)
    
    def _create_button_frame(self):
        """Create dialog button frame."""
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Buttons
        ttk.Button(
            button_frame,
            text="OK",
            command=self._on_ok,
            width=10
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            width=10
        ).pack(side=tk.RIGHT)
        
        ttk.Button(
            button_frame,
            text="Apply",
            command=self._on_apply,
            width=10
        ).pack(side=tk.RIGHT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Reset to Defaults",
            command=self._on_reset_defaults,
            width=15
        ).pack(side=tk.LEFT)
    
    def _load_current_settings(self):
        """Load current settings into the dialog."""
        try:
            # Audio settings
            self.sample_rate_var.set(self.current_config.audio.sample_rate)
            self.chunk_size_var.set(self.current_config.audio.chunk_size)
            self.vad_threshold_var.set(self.current_config.audio.vad_threshold)
            
            # Character settings
            self.intensity_var.set(self.current_config.character.intensity)
            self.max_tokens_var.set(self.current_config.character.max_tokens)
            self.temperature_var.set(self.current_config.character.temperature)
            
            # STT settings
            self.stt_model_size_var.set(self.current_config.stt.model_size)
            self.stt_device_var.set(self.current_config.stt.device)
            self.stt_language_var.set(self.current_config.stt.language)
            
            # TTS settings
            self.tts_device_var.set(self.current_config.tts.device)
            self.tts_speed_var.set(self.current_config.tts.speed)
            
            # Performance settings
            self.max_latency_var.set(self.current_config.performance.max_latency_ms)
            self.gpu_memory_var.set(self.current_config.performance.gpu_memory_fraction)
            self.enable_offloading_var.set(self.current_config.performance.enable_model_offloading)
            
            # Logging settings
            self.log_level_var.set(self.current_config.logging.level)
            
            # Update labels
            self._update_vad_label(self.vad_threshold_var.get())
            self._update_intensity_label(self.intensity_var.get())
            self._update_temperature_label(self.temperature_var.get())
            self._update_speed_label(self.tts_speed_var.get())
            self._update_gpu_label(self.gpu_memory_var.get())
            
        except Exception as e:
            logger.error(f"Failed to load current settings: {e}")
            messagebox.showerror("Settings Error", f"Failed to load current settings: {e}")
    
    def _load_audio_devices(self):
        """Load available audio devices."""
        try:
            self.input_devices, self.output_devices = get_audio_devices()
            
            # Update input device dropdown
            input_device_names = [f"{device.name} (ID: {device.id})" for device in self.input_devices]
            self.input_device_combo['values'] = input_device_names
            
            # Update output device dropdown
            output_device_names = [f"{device.name} (ID: {device.id})" for device in self.output_devices]
            self.output_device_combo['values'] = output_device_names
            
            # Set current selections
            self._set_current_device_selections()
            
        except Exception as e:
            logger.error(f"Failed to load audio devices: {e}")
            messagebox.showerror("Audio Device Error", f"Failed to load audio devices: {e}")
    
    def _set_current_device_selections(self):
        """Set current device selections in dropdowns."""
        try:
            # Set input device
            current_input_id = self.current_config.audio.input_device_id
            if current_input_id >= 0:
                for i, device in enumerate(self.input_devices):
                    if device.id == current_input_id:
                        self.input_device_var.set(self.input_device_combo['values'][i])
                        break
            
            # Set output device
            current_output_id = self.current_config.audio.output_device_id
            if current_output_id >= 0:
                for i, device in enumerate(self.output_devices):
                    if device.id == current_output_id:
                        self.output_device_var.set(self.output_device_combo['values'][i])
                        break
                        
        except Exception as e:
            logger.error(f"Failed to set current device selections: {e}")
    
    def _refresh_audio_devices(self):
        """Refresh audio device lists."""
        try:
            current_input = self.input_device_var.get()
            current_output = self.output_device_var.get()
            
            self._load_audio_devices()
            
            # Try to restore previous selections
            if current_input and current_input in self.input_device_combo['values']:
                self.input_device_var.set(current_input)
            
            if current_output and current_output in self.output_device_combo['values']:
                self.output_device_var.set(current_output)
            
            messagebox.showinfo("Refresh Complete", "Audio devices refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh audio devices: {e}")
            messagebox.showerror("Refresh Error", f"Failed to refresh audio devices: {e}")
    
    def _update_vad_label(self, value):
        """Update VAD threshold label."""
        self.vad_label.config(text=f"{float(value):.2f}")
    
    def _update_intensity_label(self, value):
        """Update intensity label."""
        self.intensity_label.config(text=f"{float(value):.2f}")
    
    def _update_temperature_label(self, value):
        """Update temperature label."""
        self.temperature_label.config(text=f"{float(value):.2f}")
    
    def _update_speed_label(self, value):
        """Update speed label."""
        self.speed_label.config(text=f"{float(value):.2f}")
    
    def _update_gpu_label(self, value):
        """Update GPU memory label."""
        self.gpu_label.config(text=f"{float(value):.2f}")
    
    def _extract_device_id(self, device_text: str) -> Optional[int]:
        """Extract device ID from device text."""
        try:
            if "(ID: " in device_text and ")" in device_text:
                start = device_text.find("(ID: ") + 5
                end = device_text.find(")", start)
                return int(device_text[start:end])
        except (ValueError, IndexError):
            pass
        return None
    
    def _create_config_from_settings(self) -> AppConfig:
        """Create AppConfig from current dialog settings."""
        try:
            # Extract device IDs
            input_device_id = self._extract_device_id(self.input_device_var.get()) or -1
            output_device_id = self._extract_device_id(self.output_device_var.get()) or -1
            
            # Create new configuration
            new_config = AppConfig(
                audio=AudioConfig(
                    sample_rate=self.sample_rate_var.get(),
                    chunk_size=self.chunk_size_var.get(),
                    input_device_id=input_device_id,
                    output_device_id=output_device_id,
                    vad_threshold=self.vad_threshold_var.get()
                ),
                stt=STTConfig(
                    model_size=self.stt_model_size_var.get(),
                    device=self.stt_device_var.get(),
                    language=self.stt_language_var.get()
                ),
                character=CharacterConfig(
                    default_character=self.current_config.character.default_character,
                    intensity=self.intensity_var.get(),
                    llm_model_path=self.current_config.character.llm_model_path,
                    max_tokens=self.max_tokens_var.get(),
                    temperature=self.temperature_var.get()
                ),
                tts=TTSConfig(
                    model_path=self.current_config.tts.model_path,
                    device=self.tts_device_var.get(),
                    sample_rate=self.current_config.tts.sample_rate,
                    speed=self.tts_speed_var.get()
                ),
                performance=PerformanceConfig(
                    max_latency_ms=self.max_latency_var.get(),
                    gpu_memory_fraction=self.gpu_memory_var.get(),
                    enable_model_offloading=self.enable_offloading_var.get(),
                    batch_size=self.current_config.performance.batch_size
                ),
                logging=LoggingConfig(
                    level=self.log_level_var.get(),
                    file=self.current_config.logging.file,
                    max_file_size=self.current_config.logging.max_file_size,
                    backup_count=self.current_config.logging.backup_count
                )
            )
            
            return new_config
            
        except Exception as e:
            logger.error(f"Failed to create config from settings: {e}")
            raise
    
    def _validate_settings(self) -> bool:
        """Validate current settings."""
        try:
            # Create config to trigger validation
            self._create_config_from_settings()
            return True
        except Exception as e:
            messagebox.showerror("Validation Error", f"Invalid settings: {e}")
            return False
    
    def _on_ok(self):
        """Handle OK button click."""
        if self._validate_settings():
            try:
                new_config = self._create_config_from_settings()
                self.config_manager.save_config(new_config)
                
                # Notify callback
                if self.on_settings_changed:
                    self.on_settings_changed(new_config)
                
                self.result = new_config
                self.dialog.destroy()
                
            except Exception as e:
                logger.error(f"Failed to save settings: {e}")
                messagebox.showerror("Save Error", f"Failed to save settings: {e}")
    
    def _on_cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()
    
    def _on_apply(self):
        """Handle Apply button click."""
        if self._validate_settings():
            try:
                new_config = self._create_config_from_settings()
                self.config_manager.save_config(new_config)
                
                # Notify callback
                if self.on_settings_changed:
                    self.on_settings_changed(new_config)
                
                # Update current config reference
                self.current_config = new_config
                
                messagebox.showinfo("Settings Applied", "Settings have been applied successfully")
                
            except Exception as e:
                logger.error(f"Failed to apply settings: {e}")
                messagebox.showerror("Apply Error", f"Failed to apply settings: {e}")
    
    def _on_reset_defaults(self):
        """Handle Reset to Defaults button click."""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            try:
                # Create default configuration
                default_config = AppConfig(
                    audio=AudioConfig(),
                    stt=STTConfig(),
                    character=CharacterConfig(),
                    tts=TTSConfig(),
                    performance=PerformanceConfig(),
                    logging=LoggingConfig()
                )
                
                # Update current config and reload settings
                self.current_config = default_config
                self._load_current_settings()
                
                messagebox.showinfo("Reset Complete", "Settings have been reset to defaults")
                
            except Exception as e:
                logger.error(f"Failed to reset settings: {e}")
                messagebox.showerror("Reset Error", f"Failed to reset settings: {e}")
    
    def show_modal(self) -> Optional[AppConfig]:
        """
        Show the dialog modally and return the result.
        
        Returns:
            New configuration if OK was clicked, None if cancelled
        """
        self.dialog.wait_window()
        return self.result