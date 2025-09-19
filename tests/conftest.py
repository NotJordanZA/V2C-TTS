"""
Pytest configuration and fixtures for the voice character transformation tests.

This module provides shared fixtures and configuration for all test modules,
including integration test setup and performance benchmarking utilities.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import AppConfig


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks (very slow)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU acceleration"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add integration marker to integration test files
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add benchmark marker to benchmark test files
        if "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.benchmark)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.nodeid for keyword in ["integration", "benchmark", "end_to_end"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config():
    """Create a test configuration with safe defaults."""
    from src.core.config import AppConfig, AudioConfig, STTConfig, CharacterConfig, TTSConfig, PerformanceConfig, LoggingConfig
    
    return AppConfig(
        audio=AudioConfig(
            sample_rate=16000,
            chunk_size=1024,
            input_device_id=0,
            output_device_id=0
        ),
        stt=STTConfig(
            model_size="base",
            device="cpu"
        ),
        character=CharacterConfig(
            profiles_dir="characters",
            default_character="default",
            llm_model_path="models/llm/test_model.bin"
        ),
        tts=TTSConfig(
            model_path="models/tts/test_model.pth",
            device="cpu",
            voice_models_dir="models/voices"
        ),
        performance=PerformanceConfig(
            max_latency_ms=2000,
            gpu_memory_fraction=0.8
        ),
        logging=LoggingConfig(
            level="INFO"
        )
    )


@pytest.fixture
def mock_audio_devices():
    """Mock audio device list for testing."""
    return [
        {"id": 0, "name": "Default Audio Device", "channels": 2, "sample_rate": 44100},
        {"id": 1, "name": "Test Microphone", "channels": 1, "sample_rate": 16000},
        {"id": 2, "name": "Test Speakers", "channels": 2, "sample_rate": 48000}
    ]


@pytest.fixture
def sample_character_profile():
    """Create a sample character profile for testing."""
    return {
        "name": "test-character",
        "description": "A test character for unit testing",
        "personality_traits": ["friendly", "helpful", "test-oriented"],
        "speech_patterns": {
            "hello": "greetings",
            "goodbye": "farewell",
            "test": "examination"
        },
        "vocabulary_preferences": {
            "greetings": ["hello", "hi", "greetings"],
            "farewells": ["goodbye", "bye", "farewell"]
        },
        "transformation_prompt": "Transform the following text as a test character: {text}",
        "voice_model_path": "models/voices/test_character.pth",
        "intensity_multiplier": 1.0
    }


@pytest.fixture
def integration_test_data():
    """Provide test data for integration tests."""
    return {
        "sample_texts": [
            "Hello, how are you today?",
            "This is a test of the voice transformation system.",
            "The weather is nice today, isn't it?",
            "Can you help me with this problem?",
            "Thank you for your assistance."
        ],
        "expected_transformations": {
            "anime-waifu": [
                "Hello, how are you today desu~?",
                "This is a test of the voice transformation system nya~",
                "The weather is nice today, isn't it kawaii~?",
                "Can you help me with this problem~?",
                "Thank you for your assistance arigatou~"
            ]
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables and cleanup."""
    # Set test environment variables
    original_env = os.environ.copy()
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during testing
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability for testing."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1):
        yield True


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU unavailability for testing CPU fallback."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.cuda.device_count', return_value=0):
        yield False


class AsyncContextManager:
    """Helper class for async context manager testing."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.entered = False
        self.exited = False
    
    async def __aenter__(self):
        self.entered = True
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False


@pytest.fixture
def async_context_manager():
    """Create an async context manager for testing."""
    return AsyncContextManager


# Performance testing utilities
@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for testing."""
    return {
        "max_latency_ms": 2000,
        "max_initialization_time_ms": 5000,
        "max_character_switch_time_ms": 5000,
        "min_success_rate": 0.95,
        "max_memory_growth_mb": 50,
        "max_concurrent_degradation_factor": 2.0
    }


# Skip markers for different test categories
def pytest_runtest_setup(item):
    """Set up individual test runs with appropriate skips."""
    # Skip GPU tests if no GPU is available (in CI environments)
    if "gpu" in item.keywords:
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")
    
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--runslow", default=False):
        pytest.skip("need --runslow option to run")
    
    # Skip benchmark tests unless explicitly requested
    if "benchmark" in item.keywords and not item.config.getoption("--benchmark", default=False):
        pytest.skip("need --benchmark option to run")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="run benchmark tests"
    )
    parser.addoption(
        "--integration", action="store_true", default=False, help="run integration tests"
    )


# Test data generators
@pytest.fixture
def audio_test_data():
    """Generate test audio data."""
    import numpy as np
    
    def generate_test_audio(duration_seconds=1.0, sample_rate=16000, frequency=440):
        """Generate sine wave test audio."""
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        return audio.astype(np.float32)
    
    return generate_test_audio


@pytest.fixture
def text_test_data():
    """Generate test text data."""
    return {
        "short_texts": [
            "Hi",
            "Yes",
            "No",
            "Thanks",
            "Hello"
        ],
        "medium_texts": [
            "How are you doing today?",
            "This is a test message.",
            "The weather looks nice outside.",
            "Can you help me with this?",
            "I appreciate your assistance."
        ],
        "long_texts": [
            "This is a longer text that should test the system's ability to handle more complex transformations and processing.",
            "The voice character transformation system needs to be able to process various lengths of text input while maintaining performance.",
            "Integration testing is crucial for ensuring that all components work together seamlessly in the complete pipeline."
        ]
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up any test files created during testing."""
    test_files = []
    
    yield test_files
    
    # Clean up any files that were created during testing
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass  # Ignore cleanup errors