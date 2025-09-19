#!/usr/bin/env python3
"""
Character transformation demo using real LLM integration.

This demo shows how to use the character transformation system with
different LLM backends (Ollama, llama.cpp, or mock).
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from character import CharacterTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_character_transformation():
    """Demonstrate character transformation with different characters."""
    
    print("🎭 Character Transformation Demo")
    print("=" * 50)
    
    # Test different LLM backends
    backends_to_test = [
        ("ollama", "llama3.2:1b"),  # Try Ollama first
        ("mock", "mock_model"),     # Fallback to mock
    ]
    
    for backend_type, model_name in backends_to_test:
        print(f"\n🔧 Testing {backend_type.upper()} backend...")
        
        try:
            # Initialize transformer with specific backend
            from character.llm_processor import create_llm_processor
            
            # Create LLM processor with specific backend
            llm_processor = create_llm_processor(model_name, processor_type=backend_type)
            
            transformer = CharacterTransformer(
                llm_model_path=model_name,
                profiles_directory="characters"
            )
            
            # Override the processor creation
            transformer.llm_processor = llm_processor
            
            async with transformer as t:
                print(f"✅ {backend_type.upper()} backend initialized successfully!")
                
                # Get available characters
                characters = t.get_available_characters()
                print(f"📚 Available characters: {', '.join(characters)}")
                
                # Test text to transform
                test_texts = [
                    "Hello, how are you doing today?",
                    "I'm really excited about this new project!",
                    "Thank you so much for your help with this."
                ]
                
                # Test different characters
                test_characters = ["default", "anime-waifu", "patriotic-american"]
                
                for char_name in test_characters:
                    if char_name not in characters:
                        continue
                        
                    print(f"\n🎭 Testing character: {char_name}")
                    print("-" * 30)
                    
                    t.set_character(char_name)
                    
                    for text in test_texts:
                        try:
                            result = await t.transform_text(text)
                            
                            print(f"Original:    {result.original_text}")
                            print(f"Transformed: {result.transformed_text}")
                            print(f"Time:        {result.processing_time_ms:.1f}ms")
                            print(f"Cached:      {result.cached}")
                            print()
                            
                        except Exception as e:
                            print(f"❌ Error transforming text: {e}")
                
                # Test caching
                print("🔄 Testing caching...")
                result1 = await t.transform_text("Hello world!")
                result2 = await t.transform_text("Hello world!")  # Should be cached
                
                print(f"First call:  {result1.processing_time_ms:.1f}ms (cached: {result1.cached})")
                print(f"Second call: {result2.processing_time_ms:.1f}ms (cached: {result2.cached})")
                
                # Show stats
                stats = t.get_transformation_stats()
                print(f"\n📊 System Stats:")
                print(f"  - Current character: {stats['current_character']}")
                print(f"  - Current intensity: {stats['current_intensity']}")
                print(f"  - Cache size: {stats['cache_stats']['size']}")
                print(f"  - LLM backend: {stats['llm_info'].get('processor_type', 'unknown')}")
                
                # Success - we found a working backend
                return
                
        except Exception as e:
            print(f"❌ {backend_type.upper()} backend failed: {e}")
            continue
    
    print("\n❌ No working LLM backends found!")
    print("\n💡 To use Ollama:")
    print("   1. Install Ollama: https://ollama.ai/")
    print("   2. Run: ollama pull llama3.2:1b")
    print("   3. Start Ollama server: ollama serve")


async def demo_intensity_levels():
    """Demonstrate different transformation intensity levels."""
    
    print("\n🎚️ Intensity Level Demo")
    print("=" * 50)
    
    try:
        async with CharacterTransformer(
            llm_model_path="llama3.2:1b",
            profiles_directory="characters"
        ) as transformer:
            
            transformer.set_character("anime-waifu")
            test_text = "I really love this new game!"
            
            intensities = [0.0, 0.5, 1.0, 1.5, 2.0]
            
            for intensity in intensities:
                result = await transformer.transform_text(
                    test_text, 
                    intensity=intensity
                )
                
                print(f"Intensity {intensity}: {result.transformed_text}")
                
    except Exception as e:
        print(f"❌ Intensity demo failed: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(demo_character_transformation())
        asyncio.run(demo_intensity_levels())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)