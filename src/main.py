"""
Main entry point for the Voice Character Transformation System.
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

from .core.config import ConfigManager
from .core.interfaces import PipelineConfig


def setup_logging(config):
    """Set up logging configuration."""
    logger.remove()  # Remove default handler
    
    # Add file handler
    log_path = Path(config.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path,
        level=config.logging.level,
        rotation=config.logging.max_file_size,
        retention=config.logging.backup_count,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=config.logging.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )


async def main():
    """Main application entry point."""
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Setup logging
        setup_logging(config)
        logger.info("Starting Voice Character Transformation System")
        
        # TODO: Initialize and start the pipeline
        # This will be implemented in later tasks
        logger.info("System initialized successfully")
        
        # Keep the application running
        logger.info("System ready. Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        logger.info("Voice Character Transformation System stopped")


if __name__ == "__main__":
    asyncio.run(main())