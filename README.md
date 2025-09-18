# Real-Time Voice Character Transformation System

A desktop application that transforms your voice in real-time to match different character personas using local AI models.

## Features

- Real-time voice capture and processing
- Local Speech-to-Text using Whisper
- Character-based text transformation using local LLM
- Text-to-Speech with character voice models
- GPU acceleration for optimal performance
- Modular architecture for easy extension

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support (RTX 4070 Super/RTX 5090 recommended)
- 16GB+ RAM
- 8GB+ VRAM

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required models (instructions coming in later tasks)

## Usage

```bash
python -m src.main
```

## Project Structure

```
├── src/
│   ├── audio/          # Audio capture and output
│   ├── stt/            # Speech-to-Text processing
│   ├── character/      # Character transformation
│   ├── tts/            # Text-to-Speech generation
│   ├── ui/             # User interface
│   ├── core/           # Core interfaces and configuration
│   └── main.py         # Application entry point
├── config/             # Configuration files
├── models/             # AI model storage
├── characters/         # Character profile definitions
├── tests/              # Unit and integration tests
└── logs/               # Application logs
```

## Development

This project follows a modular architecture with clear interfaces between components. See the design document for detailed architecture information.

## License

MIT License