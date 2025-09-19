"""
Main entry point for the Voice Character Transformation System.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Try to import loguru, fall back to standard logging if not available
try:
    from loguru import logger
    HAS_LOGURU = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    HAS_LOGURU = False

try:
    # Try relative imports first (when run as module)
    from .core.config import ConfigManager
    from .core.application import ApplicationLifecycleManager, InitializationProgress
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from core.config import ConfigManager
    from core.application import ApplicationLifecycleManager, InitializationProgress


def setup_logging(config):
    """Set up logging configuration."""
    if HAS_LOGURU:
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
    else:
        # Use standard logging
        logging.basicConfig(
            level=getattr(logging, config.logging.level, logging.INFO),
            format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            handlers=[
                logging.FileHandler(config.logging.file),
                logging.StreamHandler(sys.stderr)
            ]
        )


def progress_callback(progress: InitializationProgress):
    """Callback for initialization progress updates."""
    if progress.error:
        logger.error(f"Initialization error in {progress.stage}: {progress.error}")
    else:
        logger.info(f"Initialization: {progress.stage} ({progress.progress*100:.1f}%) - {progress.message}")


async def main():
    """Main application entry point with proper lifecycle management."""
    app_manager = None
    
    try:
        logger.info("Starting Voice Character Transformation System")
        
        # Create application lifecycle manager
        app_manager = ApplicationLifecycleManager()
        app_manager.set_progress_callback(progress_callback)
        
        # Setup interrupt handlers for graceful shutdown
        app_manager.setup_interrupt_handlers()
        
        # Initialize application
        logger.info("Initializing application components...")
        initialization_success = await app_manager.initialize_application()
        
        if not initialization_success:
            logger.error("Application initialization failed")
            return 1
        
        logger.info("Application initialized successfully")
        
        # Run application
        logger.info("Starting application main loop")
        app_manager.run_application()
        
        logger.info("Application completed normally")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        if app_manager:
            await app_manager.shutdown_application()
        return 0
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        if app_manager:
            try:
                await app_manager.shutdown_application()
            except Exception as shutdown_error:
                logger.error(f"Error during shutdown: {shutdown_error}")
        return 1
        
    finally:
        logger.info("Voice Character Transformation System stopped")


def run_application():
    """Entry point for running the application."""
    try:
        # Setup basic logging before config is loaded
        if HAS_LOGURU:
            logger.remove()
            logger.add(sys.stderr, level="INFO", 
                      format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(levelname)s | %(message)s',
                handlers=[logging.StreamHandler(sys.stderr)]
            )
        
        # Run the async main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except Exception as e:
        if HAS_LOGURU:
            logger.error(f"Fatal application error: {e}")
        else:
            logging.error(f"Fatal application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_application()