"""
Main application window for the Real-Time Voice Character Transformation system.

This module provides the main GUI interface using Tkinter, including character
selection, audio device configuration, and pipeline control.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Callable
import logging
from pathlib import Path

try:
    # Try relative imports first (when used as a package)
    from ..audio.device_manager import AudioDeviceManager, get_audio_devices, get_default_devices
    from ..character.profile import CharacterProfileManager
    from ..core.config import ConfigManager, AppConfig
    from ..core.interfaces import AudioDevice
except ImportError:
    # Fallback to absolute imports (when used as a script)
    from audio.device_manager import AudioDeviceManager, get_audio_devices, get_default_devices
    from character.profile import CharacterProfileManager
    from core.config import ConfigManager, AppConfig
    from core.interfaces import AudioDevice

logger = logging.getLogger(__name__)


class MainWindow:
    """Main application window for voice character transformation."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the main window.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # Initialize managers
        self.character_manager = CharacterProfileManager()
        
        # Initialize UI components
        self.root = tk.Tk()
        self.root.title("Voice Character Transformation")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables for UI controls
        self.selected_character = tk.StringVar()
        self.selected_input_device = tk.StringVar()
        self.selected_output_device = tk.StringVar()
        
        # Audio device lists
        self.input_devices: List[AudioDevice] = []
        self.output_devices: List[AudioDevice] = []
        
        # Callbacks for external control
        self.on_character_changed: Optional[Callable[[str], None]] = None
        self.on_input_device_changed: Optional[Callable[[int], None]] = None
        self.on_output_device_changed: Optional[Callable[[int], None]] = None
        self.on_start_pipeline: Optional[Callable[[], None]] = None
        self.on_stop_pipeline: Optional[Callable[[], None]] = None
        
        # Initialize UI
        self._setup_ui()
        self._load_initial_data()
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def _setup_ui(self):
        """Set up the user interface components."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Character selection section
        self._create_character_section(main_frame, row=0)
        
        # Audio device selection section
        self._create_audio_section(main_frame, row=1)
        
        # Control buttons section
        self._create_control_section(main_frame, row=2)
        
        # Status section (placeholder for subtask 8.2)
        self._create_status_section(main_frame, row=3)
    
    def _create_character_section(self, parent: ttk.Frame, row: int):
        """Create character selection section."""
        # Character selection frame
        char_frame = ttk.LabelFrame(parent, text="Character Selection", padding="10")
        char_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        char_frame.columnconfigure(1, weight=1)
        
        # Character dropdown
        ttk.Label(char_frame, text="Character:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.character_combo = ttk.Combobox(
            char_frame, 
            textvariable=self.selected_character,
            state="readonly",
            width=30
        )
        self.character_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Character description
        ttk.Label(char_frame, text="Description:").grid(row=1, column=0, sticky=(tk.W, tk.N), padx=(0, 10), pady=(5, 0))
        self.character_description = tk.Text(
            char_frame, 
            height=3, 
            width=50,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("TkDefaultFont", 9)
        )
        self.character_description.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Scrollbar for description
        desc_scrollbar = ttk.Scrollbar(char_frame, orient=tk.VERTICAL, command=self.character_description.yview)
        desc_scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S), pady=(5, 0))
        self.character_description.configure(yscrollcommand=desc_scrollbar.set)
    
    def _create_audio_section(self, parent: ttk.Frame, row: int):
        """Create audio device selection section."""
        # Audio devices frame
        audio_frame = ttk.LabelFrame(parent, text="Audio Devices", padding="10")
        audio_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        audio_frame.columnconfigure(1, weight=1)
        
        # Input device selection
        ttk.Label(audio_frame, text="Input Device:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.input_device_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.selected_input_device,
            state="readonly",
            width=40
        )
        self.input_device_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Refresh input devices button
        ttk.Button(
            audio_frame,
            text="Refresh",
            command=self._refresh_input_devices,
            width=10
        ).grid(row=0, column=2, padx=(5, 0))
        
        # Output device selection
        ttk.Label(audio_frame, text="Output Device:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.output_device_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.selected_output_device,
            state="readonly",
            width=40
        )
        self.output_device_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(5, 0))
        
        # Refresh output devices button
        ttk.Button(
            audio_frame,
            text="Refresh",
            command=self._refresh_output_devices,
            width=10
        ).grid(row=1, column=2, padx=(5, 0), pady=(5, 0))
    
    def _create_control_section(self, parent: ttk.Frame, row: int):
        """Create pipeline control section."""
        # Control buttons frame
        control_frame = ttk.LabelFrame(parent, text="Pipeline Control", padding="10")
        control_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start/Stop buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(anchor=tk.W)
        
        self.start_button = ttk.Button(
            button_frame,
            text="Start Pipeline",
            command=self._on_start_clicked,
            width=15
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Pipeline",
            command=self._on_stop_clicked,
            state=tk.DISABLED,
            width=15
        )
        self.stop_button.pack(side=tk.LEFT)
        
        # Settings button
        ttk.Button(
            button_frame,
            text="Settings",
            command=self._open_settings,
            width=15
        ).pack(side=tk.LEFT, padx=(20, 0))
    
    def _create_status_section(self, parent: ttk.Frame, row: int):
        """Create status display section with real-time indicators."""
        # Status frame
        status_frame = ttk.LabelFrame(parent, text="Real-Time Status", padding="10")
        status_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        status_frame.rowconfigure(2, weight=1)  # Make text areas expandable
        
        # Audio input level indicator
        self._create_audio_level_indicator(status_frame, row=0)
        
        # Pipeline stage indicators
        self._create_pipeline_indicators(status_frame, row=1)
        
        # Text display areas
        self._create_text_displays(status_frame, row=2)
        
        # Error message display
        self._create_error_display(status_frame, row=3)
    
    def _create_audio_level_indicator(self, parent: ttk.Frame, row: int):
        """Create audio input level indicator."""
        # Audio level frame
        audio_frame = ttk.Frame(parent)
        audio_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        audio_frame.columnconfigure(1, weight=1)
        
        # Label
        ttk.Label(audio_frame, text="Microphone Level:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        # Progress bar for audio level
        self.audio_level_var = tk.DoubleVar()
        self.audio_level_bar = ttk.Progressbar(
            audio_frame,
            variable=self.audio_level_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        self.audio_level_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Level percentage label
        self.audio_level_label = ttk.Label(audio_frame, text="0%")
        self.audio_level_label.grid(row=0, column=2, sticky=tk.W)
    
    def _create_pipeline_indicators(self, parent: ttk.Frame, row: int):
        """Create pipeline stage progress indicators."""
        # Pipeline frame
        pipeline_frame = ttk.LabelFrame(parent, text="Pipeline Stages", padding="5")
        pipeline_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        pipeline_frame.columnconfigure(1, weight=1)
        
        # Pipeline stages
        self.pipeline_stages = {
            'audio_capture': {'name': 'Audio Capture', 'status': 'idle'},
            'speech_to_text': {'name': 'Speech-to-Text', 'status': 'idle'},
            'character_transform': {'name': 'Character Transform', 'status': 'idle'},
            'text_to_speech': {'name': 'Text-to-Speech', 'status': 'idle'},
            'audio_output': {'name': 'Audio Output', 'status': 'idle'}
        }
        
        # Create indicators for each stage
        self.stage_indicators = {}
        for i, (stage_key, stage_info) in enumerate(self.pipeline_stages.items()):
            # Stage name label
            stage_label = ttk.Label(pipeline_frame, text=f"{stage_info['name']}:")
            stage_label.grid(row=i, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            
            # Status indicator (colored circle using Unicode)
            status_label = ttk.Label(pipeline_frame, text="●", foreground="gray", font=("TkDefaultFont", 12))
            status_label.grid(row=i, column=1, sticky=tk.W, padx=(0, 5), pady=2)
            
            # Status text
            status_text = ttk.Label(pipeline_frame, text="Idle", font=("TkDefaultFont", 9))
            status_text.grid(row=i, column=2, sticky=tk.W, pady=2)
            
            # Store references
            self.stage_indicators[stage_key] = {
                'indicator': status_label,
                'text': status_text
            }
    
    def _create_text_displays(self, parent: ttk.Frame, row: int):
        """Create text display areas for original and transformed text."""
        # Text display frame
        text_frame = ttk.LabelFrame(parent, text="Text Processing", padding="5")
        text_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(1, weight=1)
        text_frame.rowconfigure(1, weight=1)
        
        # Original text section
        ttk.Label(text_frame, text="Original Text:", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5), pady=(0, 5)
        )
        
        self.original_text = tk.Text(
            text_frame,
            height=6,
            width=40,
            wrap=tk.WORD,
            font=("TkDefaultFont", 9),
            state=tk.DISABLED,
            bg="#f8f8f8"
        )
        self.original_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Original text scrollbar
        orig_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.original_text.yview)
        orig_scrollbar.grid(row=1, column=0, sticky=(tk.E, tk.N, tk.S), padx=(0, 5))
        self.original_text.configure(yscrollcommand=orig_scrollbar.set)
        
        # Transformed text section
        ttk.Label(text_frame, text="Transformed Text:", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=1, sticky=tk.W, padx=(5, 0), pady=(0, 5)
        )
        
        self.transformed_text = tk.Text(
            text_frame,
            height=6,
            width=40,
            wrap=tk.WORD,
            font=("TkDefaultFont", 9),
            state=tk.DISABLED,
            bg="#f0f8ff"
        )
        self.transformed_text.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Transformed text scrollbar
        trans_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.transformed_text.yview)
        trans_scrollbar.grid(row=1, column=1, sticky=(tk.E, tk.N, tk.S), padx=(5, 0))
        self.transformed_text.configure(yscrollcommand=trans_scrollbar.set)
    
    def _create_error_display(self, parent: ttk.Frame, row: int):
        """Create error message display area."""
        # Error display frame
        error_frame = ttk.LabelFrame(parent, text="Messages", padding="5")
        error_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        error_frame.columnconfigure(0, weight=1)
        
        # Error message text area
        self.error_text = tk.Text(
            error_frame,
            height=3,
            wrap=tk.WORD,
            font=("TkDefaultFont", 9),
            state=tk.DISABLED,
            bg="#fff8f8"
        )
        self.error_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Error text scrollbar
        error_scrollbar = ttk.Scrollbar(error_frame, orient=tk.VERTICAL, command=self.error_text.yview)
        error_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.error_text.configure(yscrollcommand=error_scrollbar.set)
        
        # Clear messages button
        ttk.Button(
            error_frame,
            text="Clear Messages",
            command=self._clear_messages,
            width=15
        ).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
    
    def _setup_event_handlers(self):
        """Set up event handlers for UI controls."""
        # Character selection change
        self.selected_character.trace_add('write', self._on_character_selection_changed)
        
        # Audio device selection changes
        self.selected_input_device.trace_add('write', self._on_input_device_selection_changed)
        self.selected_output_device.trace_add('write', self._on_output_device_selection_changed)
        
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_closing)
    
    def _load_initial_data(self):
        """Load initial data for dropdowns and selections."""
        try:
            # Load character profiles
            self._load_character_profiles()
            
            # Load audio devices
            self._load_audio_devices()
            
            # Set initial selections from config
            self._set_initial_selections()
            
        except Exception as e:
            logger.error(f"Failed to load initial data: {e}")
            messagebox.showerror("Initialization Error", f"Failed to load initial data: {e}")
    
    def _load_character_profiles(self):
        """Load available character profiles."""
        try:
            available_characters = self.character_manager.get_available_profiles()
            self.character_combo['values'] = available_characters
            
            if available_characters:
                # Set default character from config or first available
                default_char = self.config.character.default_character
                if default_char in available_characters:
                    self.selected_character.set(default_char)
                else:
                    self.selected_character.set(available_characters[0])
            
            logger.info(f"Loaded {len(available_characters)} character profiles")
            
        except Exception as e:
            logger.error(f"Failed to load character profiles: {e}")
            messagebox.showerror("Character Loading Error", f"Failed to load character profiles: {e}")
    
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
            
            logger.info(f"Loaded {len(self.input_devices)} input devices and {len(self.output_devices)} output devices")
            
        except Exception as e:
            logger.error(f"Failed to load audio devices: {e}")
            messagebox.showerror("Audio Device Error", f"Failed to load audio devices: {e}")
    
    def _set_initial_selections(self):
        """Set initial selections based on config or defaults."""
        try:
            # Set default audio devices
            default_input, default_output = get_default_devices()
            
            # Set input device
            if self.config.audio.input_device_id >= 0:
                # Use configured device
                for i, device in enumerate(self.input_devices):
                    if device.id == self.config.audio.input_device_id:
                        self.selected_input_device.set(self.input_device_combo['values'][i])
                        break
            elif default_input:
                # Use system default
                for i, device in enumerate(self.input_devices):
                    if device.id == default_input.id:
                        self.selected_input_device.set(self.input_device_combo['values'][i])
                        break
            elif self.input_devices:
                # Use first available
                self.selected_input_device.set(self.input_device_combo['values'][0])
            
            # Set output device
            if self.config.audio.output_device_id >= 0:
                # Use configured device
                for i, device in enumerate(self.output_devices):
                    if device.id == self.config.audio.output_device_id:
                        self.selected_output_device.set(self.output_device_combo['values'][i])
                        break
            elif default_output:
                # Use system default
                for i, device in enumerate(self.output_devices):
                    if device.id == default_output.id:
                        self.selected_output_device.set(self.output_device_combo['values'][i])
                        break
            elif self.output_devices:
                # Use first available
                self.selected_output_device.set(self.output_device_combo['values'][0])
            
        except Exception as e:
            logger.error(f"Failed to set initial selections: {e}")
    
    def _on_character_selection_changed(self, *args):
        """Handle character selection change."""
        try:
            character_name = self.selected_character.get()
            if character_name:
                # Load character profile and update description
                profile = self.character_manager.load_profile(character_name)
                
                # Update description text
                self.character_description.config(state=tk.NORMAL)
                self.character_description.delete(1.0, tk.END)
                self.character_description.insert(1.0, profile.description)
                self.character_description.config(state=tk.DISABLED)
                
                # Notify external callback
                if self.on_character_changed:
                    self.on_character_changed(character_name)
                
                logger.info(f"Selected character: {character_name}")
                
        except Exception as e:
            logger.error(f"Failed to handle character selection: {e}")
            messagebox.showerror("Character Selection Error", f"Failed to load character: {e}")
    
    def _on_input_device_selection_changed(self, *args):
        """Handle input device selection change."""
        try:
            device_text = self.selected_input_device.get()
            if device_text:
                # Extract device ID from the text
                device_id = self._extract_device_id(device_text)
                if device_id is not None:
                    # Notify external callback
                    if self.on_input_device_changed:
                        self.on_input_device_changed(device_id)
                    
                    logger.info(f"Selected input device: {device_text}")
                
        except Exception as e:
            logger.error(f"Failed to handle input device selection: {e}")
    
    def _on_output_device_selection_changed(self, *args):
        """Handle output device selection change."""
        try:
            device_text = self.selected_output_device.get()
            if device_text:
                # Extract device ID from the text
                device_id = self._extract_device_id(device_text)
                if device_id is not None:
                    # Notify external callback
                    if self.on_output_device_changed:
                        self.on_output_device_changed(device_id)
                    
                    logger.info(f"Selected output device: {device_text}")
                
        except Exception as e:
            logger.error(f"Failed to handle output device selection: {e}")
    
    def _extract_device_id(self, device_text: str) -> Optional[int]:
        """Extract device ID from device text."""
        try:
            # Format is "Device Name (ID: X)"
            if "(ID: " in device_text and ")" in device_text:
                start = device_text.find("(ID: ") + 5
                end = device_text.find(")", start)
                return int(device_text[start:end])
        except (ValueError, IndexError):
            pass
        return None
    
    def _refresh_input_devices(self):
        """Refresh input device list."""
        try:
            current_selection = self.selected_input_device.get()
            self._load_audio_devices()
            
            # Try to restore previous selection
            if current_selection and current_selection in self.input_device_combo['values']:
                self.selected_input_device.set(current_selection)
            
            messagebox.showinfo("Refresh Complete", "Input devices refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh input devices: {e}")
            messagebox.showerror("Refresh Error", f"Failed to refresh input devices: {e}")
    
    def _refresh_output_devices(self):
        """Refresh output device list."""
        try:
            current_selection = self.selected_output_device.get()
            self._load_audio_devices()
            
            # Try to restore previous selection
            if current_selection and current_selection in self.output_device_combo['values']:
                self.selected_output_device.set(current_selection)
            
            messagebox.showinfo("Refresh Complete", "Output devices refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh output devices: {e}")
            messagebox.showerror("Refresh Error", f"Failed to refresh output devices: {e}")
    
    def _on_start_clicked(self):
        """Handle start pipeline button click."""
        try:
            # Validate selections
            if not self.selected_character.get():
                messagebox.showerror("Validation Error", "Please select a character")
                return
            
            if not self.selected_input_device.get():
                messagebox.showerror("Validation Error", "Please select an input device")
                return
            
            if not self.selected_output_device.get():
                messagebox.showerror("Validation Error", "Please select an output device")
                return
            
            # Update button states
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Notify external callback
            if self.on_start_pipeline:
                self.on_start_pipeline()
            
            logger.info("Pipeline start requested")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            messagebox.showerror("Start Error", f"Failed to start pipeline: {e}")
            # Reset button states on error
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def _on_stop_clicked(self):
        """Handle stop pipeline button click."""
        try:
            # Update button states
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Notify external callback
            if self.on_stop_pipeline:
                self.on_stop_pipeline()
            
            logger.info("Pipeline stop requested")
            
        except Exception as e:
            logger.error(f"Failed to stop pipeline: {e}")
            messagebox.showerror("Stop Error", f"Failed to stop pipeline: {e}")
    
    def _open_settings(self):
        """Open settings dialog."""
        try:
            # Import settings dialog
            try:
                from .settings_dialog import SettingsDialog
            except ImportError:
                from settings_dialog import SettingsDialog
            
            # Create and show settings dialog
            settings_dialog = SettingsDialog(self.root, self.config_manager, self.config)
            
            # Set up callback for settings changes
            def on_settings_changed(new_config):
                self.config = new_config
                logger.info("Settings updated from dialog")
                self.show_info_message("Settings updated successfully")
            
            settings_dialog.on_settings_changed = on_settings_changed
            
            # Show dialog modally
            result = settings_dialog.show_modal()
            
            if result:
                # Settings were saved, update our config reference
                self.config = result
                logger.info("Settings dialog completed with changes")
            else:
                logger.info("Settings dialog cancelled")
                
        except Exception as e:
            logger.error(f"Failed to open settings dialog: {e}")
            messagebox.showerror("Settings Error", f"Failed to open settings dialog: {e}")
    
    def _on_window_closing(self):
        """Handle window closing event."""
        try:
            # Stop pipeline if running
            if self.stop_button['state'] == tk.NORMAL:
                if self.on_stop_pipeline:
                    self.on_stop_pipeline()
            
            # Save current configuration
            self._save_current_config()
            
            # Close window
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error during window closing: {e}")
            self.root.destroy()
    
    def _save_current_config(self):
        """Save current UI selections to configuration."""
        try:
            # Update config with current selections
            if self.selected_character.get():
                self.config.character.default_character = self.selected_character.get()
            
            input_device_id = self._extract_device_id(self.selected_input_device.get())
            if input_device_id is not None:
                self.config.audio.input_device_id = input_device_id
            
            output_device_id = self._extract_device_id(self.selected_output_device.get())
            if output_device_id is not None:
                self.config.audio.output_device_id = output_device_id
            
            # Save configuration
            self.config_manager.save_config(self.config)
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def run(self):
        """Start the main event loop."""
        try:
            logger.info("Starting main window")
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise
    
    def set_pipeline_status(self, running: bool):
        """Update pipeline status and button states."""
        if running:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def get_selected_character(self) -> Optional[str]:
        """Get currently selected character name."""
        return self.selected_character.get() or None
    
    def get_selected_input_device_id(self) -> Optional[int]:
        """Get currently selected input device ID."""
        return self._extract_device_id(self.selected_input_device.get())
    
    def get_selected_output_device_id(self) -> Optional[int]:
        """Get currently selected output device ID."""
        return self._extract_device_id(self.selected_output_device.get())
    
    # Status display update methods
    
    def update_audio_level(self, level: float):
        """
        Update microphone input level indicator.
        
        Args:
            level: Audio level as percentage (0.0 to 100.0)
        """
        try:
            # Clamp level to valid range
            level = max(0.0, min(100.0, level))
            
            # Update progress bar
            self.audio_level_var.set(level)
            
            # Update percentage label
            self.audio_level_label.config(text=f"{level:.1f}%")
            
            # Color coding based on level
            if level > 80:
                # High level - red
                self.audio_level_bar.config(style="red.Horizontal.TProgressbar")
            elif level > 60:
                # Medium level - yellow
                self.audio_level_bar.config(style="yellow.Horizontal.TProgressbar")
            else:
                # Normal level - green
                self.audio_level_bar.config(style="green.Horizontal.TProgressbar")
                
        except Exception as e:
            logger.error(f"Failed to update audio level: {e}")
    
    def update_pipeline_stage(self, stage: str, status: str, message: str = ""):
        """
        Update pipeline stage status indicator.
        
        Args:
            stage: Pipeline stage name ('audio_capture', 'speech_to_text', etc.)
            status: Status ('idle', 'processing', 'complete', 'error')
            message: Optional status message
        """
        try:
            if stage not in self.stage_indicators:
                logger.warning(f"Unknown pipeline stage: {stage}")
                return
            
            indicator = self.stage_indicators[stage]['indicator']
            text_label = self.stage_indicators[stage]['text']
            
            # Update status in internal tracking
            self.pipeline_stages[stage]['status'] = status
            
            # Color and text based on status
            if status == 'idle':
                indicator.config(foreground="gray")
                text_label.config(text="Idle")
            elif status == 'processing':
                indicator.config(foreground="blue")
                text_label.config(text="Processing...")
            elif status == 'complete':
                indicator.config(foreground="green")
                text_label.config(text="Complete")
            elif status == 'error':
                indicator.config(foreground="red")
                text_label.config(text="Error")
            else:
                indicator.config(foreground="gray")
                text_label.config(text=status.title())
            
            # Add message if provided
            if message:
                current_text = text_label.cget("text")
                text_label.config(text=f"{current_text} - {message}")
                
        except Exception as e:
            logger.error(f"Failed to update pipeline stage {stage}: {e}")
    
    def update_original_text(self, text: str):
        """
        Update original text display.
        
        Args:
            text: Original text from speech-to-text
        """
        try:
            self.original_text.config(state=tk.NORMAL)
            self.original_text.delete(1.0, tk.END)
            self.original_text.insert(1.0, text)
            self.original_text.config(state=tk.DISABLED)
            
            # Auto-scroll to bottom
            self.original_text.see(tk.END)
            
        except Exception as e:
            logger.error(f"Failed to update original text: {e}")
    
    def update_transformed_text(self, text: str):
        """
        Update transformed text display.
        
        Args:
            text: Transformed text from character processing
        """
        try:
            self.transformed_text.config(state=tk.NORMAL)
            self.transformed_text.delete(1.0, tk.END)
            self.transformed_text.insert(1.0, text)
            self.transformed_text.config(state=tk.DISABLED)
            
            # Auto-scroll to bottom
            self.transformed_text.see(tk.END)
            
        except Exception as e:
            logger.error(f"Failed to update transformed text: {e}")
    
    def append_original_text(self, text: str):
        """
        Append text to original text display.
        
        Args:
            text: Text to append
        """
        try:
            self.original_text.config(state=tk.NORMAL)
            self.original_text.insert(tk.END, f"\n{text}")
            self.original_text.config(state=tk.DISABLED)
            
            # Auto-scroll to bottom
            self.original_text.see(tk.END)
            
        except Exception as e:
            logger.error(f"Failed to append original text: {e}")
    
    def append_transformed_text(self, text: str):
        """
        Append text to transformed text display.
        
        Args:
            text: Text to append
        """
        try:
            self.transformed_text.config(state=tk.NORMAL)
            self.transformed_text.insert(tk.END, f"\n{text}")
            self.transformed_text.config(state=tk.DISABLED)
            
            # Auto-scroll to bottom
            self.transformed_text.see(tk.END)
            
        except Exception as e:
            logger.error(f"Failed to append transformed text: {e}")
    
    def show_error_message(self, message: str, error_type: str = "Error"):
        """
        Display error message in the error display area.
        
        Args:
            message: Error message to display
            error_type: Type of error (Error, Warning, Info)
        """
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Format message with timestamp and type
            formatted_message = f"[{timestamp}] {error_type}: {message}\n"
            
            self.error_text.config(state=tk.NORMAL)
            self.error_text.insert(tk.END, formatted_message)
            self.error_text.config(state=tk.DISABLED)
            
            # Auto-scroll to bottom
            self.error_text.see(tk.END)
            
            # Color coding based on error type
            if error_type.lower() == "error":
                self.error_text.config(bg="#ffe6e6")  # Light red
            elif error_type.lower() == "warning":
                self.error_text.config(bg="#fff8e6")  # Light yellow
            else:
                self.error_text.config(bg="#e6f3ff")  # Light blue
                
        except Exception as e:
            logger.error(f"Failed to show error message: {e}")
    
    def show_info_message(self, message: str):
        """
        Display info message in the error display area.
        
        Args:
            message: Info message to display
        """
        self.show_error_message(message, "Info")
    
    def show_warning_message(self, message: str):
        """
        Display warning message in the error display area.
        
        Args:
            message: Warning message to display
        """
        self.show_error_message(message, "Warning")
    
    def _clear_messages(self):
        """Clear all messages from the error display."""
        try:
            self.error_text.config(state=tk.NORMAL)
            self.error_text.delete(1.0, tk.END)
            self.error_text.config(state=tk.DISABLED)
            
            # Reset background color
            self.error_text.config(bg="#fff8f8")
            
        except Exception as e:
            logger.error(f"Failed to clear messages: {e}")
    
    def reset_pipeline_status(self):
        """Reset all pipeline stage indicators to idle state."""
        try:
            for stage in self.pipeline_stages.keys():
                self.update_pipeline_stage(stage, 'idle')
                
        except Exception as e:
            logger.error(f"Failed to reset pipeline status: {e}")
    
    def clear_text_displays(self):
        """Clear both original and transformed text displays."""
        try:
            self.original_text.config(state=tk.NORMAL)
            self.original_text.delete(1.0, tk.END)
            self.original_text.config(state=tk.DISABLED)
            
            self.transformed_text.config(state=tk.NORMAL)
            self.transformed_text.delete(1.0, tk.END)
            self.transformed_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Failed to clear text displays: {e}")