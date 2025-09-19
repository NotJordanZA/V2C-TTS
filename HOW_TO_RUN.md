# How to Run the Voice Character Transformation Application

This guide explains the different ways to run the voice character transformation system, from simple demos to the full application.

## Prerequisites

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (Linux/macOS)
# For audio support
sudo apt-get install portaudio19-dev python3-pyaudio  # Ubuntu/Debian
brew install portaudio  # macOS

# For FFmpeg (audio processing)
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg  # macOS
```

### 2. Set up Models Directory

```bash
# Create model directories
mkdir -p models/voices
mkdir -p models/llm
mkdir -p models/tts

# The application will work with mock models for testing
# For production, you'll need to download actual models
```

### 3. Verify Setup

```bash
# Run the environment validation script
python scripts/validate_integration_tests.py
```

## Running Options

### 1. 🎬 Complete UI Demo (Recommended for First Run)

The easiest way to see the full application in action:

```bash
python ui_complete_demo.py
```

Or use the launcher:

```bash
python run_app.py ui
```

**What this shows:**
- Complete graphical user interface
- Character selection (anime-waifu, patriotic-american, slurring-drunk, default)
- Audio device selection
- Real-time status displays
- Settings configuration
- Simulated pipeline activity

**Features demonstrated:**
- Character switching
- Audio level monitoring
- Pipeline stage indicators
- Text transformation display
- Error handling and messaging

### 2. 🚀 Interactive Launcher (Recommended)

Use the interactive launcher to choose what to run:

```bash
python run_app.py
```

This will show a menu with options:
1. UI Demo
2. Main Application  
3. Lifecycle Demo
4. Environment Validation
5. Exit

Or run directly with arguments:

```bash
python run_app.py ui          # UI Demo
python run_app.py main        # Main Application
python run_app.py lifecycle   # Lifecycle Demo
python run_app.py validate    # Environment Check
```

### 3. 🧪 Simple System Demo

Test core systems without complex lifecycle management:

```bash
python simple_demo.py
```

**What this shows:**
- Configuration system loading
- Audio device detection
- Character profile loading
- Performance profiling
- Quality management system

### 3. 🎭 Application Lifecycle Demo

See how the application initializes and shuts down:

```bash
python examples/application_lifecycle_demo.py
```

**What this demonstrates:**
- Application startup process
- Component initialization with progress tracking
- Error handling during initialization
- Graceful shutdown procedures
- Cleanup task execution

### 4. 🎵 Individual Component Demos

Test specific components individually:

#### Audio System Demo
```bash
python examples/audio_demo.py
```
- Tests audio capture and playback
- Shows available audio devices
- Demonstrates audio processing

#### Speech-to-Text Demo
```bash
python examples/stt_demo.py
# or for a simpler version
python examples/stt_demo_simple.py
```
- Tests speech recognition
- Shows transcription accuracy
- Demonstrates different model sizes

#### Character Transformation Demo
```bash
python examples/character_demo.py
```
- Tests character profile loading
- Shows text transformation examples
- Demonstrates different character personalities

#### Voice Model Demo
```bash
python examples/voice_model_demo.py
```
- Tests text-to-speech synthesis
- Shows voice model loading
- Demonstrates character voice mapping

#### Configuration Demo
```bash
python examples/config_demo.py
```
- Shows configuration loading and validation
- Demonstrates different config formats
- Tests configuration error handling

### 5. 🧪 Testing and Validation

#### Run Integration Tests
```bash
# Run all integration tests
python scripts/run_integration_tests.py --test-type all

# Run specific test types
python scripts/run_integration_tests.py --test-type integration
python scripts/run_integration_tests.py --test-type performance
python scripts/run_integration_tests.py --test-type config
```

#### Run Performance Benchmarks
```bash
python scripts/run_integration_tests.py --test-type performance --verbose
```

#### Validate Environment
```bash
python scripts/validate_integration_tests.py
```

## Configuration

### Default Configuration

The application uses configuration files in the `config/` directory:

```bash
config/
├── default_config.yaml    # Default settings
├── user_config.yaml      # User overrides
└── user_config.json      # Alternative JSON format
```

### Character Profiles

Character profiles are stored in the `characters/` directory:

```bash
characters/
├── anime-waifu.json
├── patriotic-american.json
├── slurring-drunk.json
└── default.json
```

### Environment Variables

You can override settings with environment variables:

```bash
export AUDIO_SAMPLE_RATE=16000
export STT_MODEL_SIZE=base
export TTS_DEVICE=cpu
export LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

#### 1. Audio Device Not Found
```bash
# List available audio devices
python examples/audio_demo.py
```

#### 2. Model Loading Errors
```bash
# Check model directories exist
ls -la models/
# For testing, the app works with mock models
```

#### 3. Configuration Errors
```bash
# Validate configuration
python examples/config_demo.py
```

#### 4. Import Errors
```bash
# Ensure you're in the project root directory
pwd
# Should show the project root with src/, tests/, etc.

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Debug Mode

Run with debug logging:

```bash
export LOG_LEVEL=DEBUG
python src/main.py
```

### Performance Issues

Check system performance:

```bash
# Run performance benchmarks
python scripts/run_integration_tests.py --test-type performance

# Monitor system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"
```

## Development Workflow

### 1. Start with UI Demo
```bash
python ui_complete_demo.py
```

### 2. Test Individual Components
```bash
python examples/audio_demo.py
python examples/stt_demo.py
python examples/character_demo.py
```

### 3. Run Integration Tests
```bash
python scripts/run_integration_tests.py --test-type integration
```

### 4. Run Full Application
```bash
python src/main.py
```

## Production Deployment

### 1. Install Production Models

```bash
# Download actual models (examples)
# STT Model (Whisper)
wget https://example.com/whisper-base.pt -O models/stt/whisper-base.pt

# TTS Models
wget https://example.com/tts-model.pth -O models/tts/tts-model.pth

# Character Voice Models
wget https://example.com/anime-voice.pth -O models/voices/anime-voice.pth
```

### 2. Configure for Production

```yaml
# config/production_config.yaml
audio:
  sample_rate: 16000
  chunk_size: 1024

stt:
  model_size: "base"
  device: "cuda"  # Use GPU if available

performance:
  max_latency_ms: 2000
  gpu_memory_fraction: 0.8

logging:
  level: "INFO"
  file: "logs/production.log"
```

### 3. Run with Production Config

```bash
export CONFIG_FILE=config/production_config.yaml
python src/main.py
```

## Quick Start Summary

**For first-time users:**
1. `python scripts/validate_integration_tests.py` - Verify setup
2. `python ui_complete_demo.py` - See the full UI in action
3. `python examples/application_lifecycle_demo.py` - Understand the lifecycle
4. `python src/main.py` - Run the full application

**For developers:**
1. `python examples/audio_demo.py` - Test audio system
2. `python examples/character_demo.py` - Test character system
3. `python scripts/run_integration_tests.py --test-type all` - Run all tests
4. `python src/main.py` - Run full application

**For production:**
1. Install production models
2. Configure `config/production_config.yaml`
3. `python src/main.py` with production settings

## Support

If you encounter issues:

1. Check the logs in `logs/` directory
2. Run `python scripts/validate_integration_tests.py`
3. Try the individual component demos to isolate issues
4. Check the troubleshooting section above

The application includes comprehensive error handling and will provide detailed error messages to help diagnose issues.