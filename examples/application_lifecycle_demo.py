#!/usr/bin/env python3
"""
Demo script showing application lifecycle management.

This script demonstrates the complete application startup, initialization,
and shutdown process with proper error handling and progress reporting.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.application import ApplicationLifecycleManager, InitializationProgress


def setup_demo_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/lifecycle_demo.log')
        ]
    )


def progress_callback(progress: InitializationProgress):
    """Callback to display initialization progress."""
    if progress.error:
        print(f"❌ ERROR in {progress.stage}: {progress.error}")
    else:
        progress_bar = "█" * int(progress.progress * 20)
        progress_bar += "░" * (20 - len(progress_bar))
        print(f"🔄 [{progress_bar}] {progress.progress*100:5.1f}% - {progress.stage}: {progress.message}")


async def demo_successful_lifecycle():
    """Demonstrate successful application lifecycle."""
    print("\n" + "="*60)
    print("🚀 DEMO: Successful Application Lifecycle")
    print("="*60)
    
    app_manager = ApplicationLifecycleManager()
    app_manager.set_progress_callback(progress_callback)
    
    try:
        print("\n📋 Phase 1: Application Initialization")
        print("-" * 40)
        
        # This would normally initialize all components
        # For demo purposes, we'll mock the heavy components
        success = await app_manager.initialize_application()
        
        if success:
            print(f"\n✅ Application initialized successfully!")
            print(f"   State: {app_manager.get_application_state().value}")
            
            # Show system status
            status = app_manager.get_system_status()
            print(f"   Initialization complete: {status['initialization_complete']}")
            print(f"   Pipeline available: {status['pipeline_available']}")
            print(f"   UI available: {status['ui_available']}")
            
        else:
            print(f"\n❌ Application initialization failed!")
            return
        
        print("\n📋 Phase 2: Application Shutdown")
        print("-" * 40)
        
        # Demonstrate graceful shutdown
        await app_manager.shutdown_application()
        print(f"✅ Application shutdown completed!")
        print(f"   Final state: {app_manager.get_application_state().value}")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        # Ensure cleanup even on error
        try:
            await app_manager.emergency_shutdown()
        except:
            pass


async def demo_initialization_error():
    """Demonstrate initialization error handling."""
    print("\n" + "="*60)
    print("⚠️  DEMO: Initialization Error Handling")
    print("="*60)
    
    app_manager = ApplicationLifecycleManager()
    app_manager.set_progress_callback(progress_callback)
    
    try:
        print("\n📋 Simulating initialization error...")
        print("-" * 40)
        
        # Force an error during configuration loading
        original_init_config = app_manager._initialize_configuration
        
        async def failing_config():
            raise Exception("Simulated configuration error")
        
        app_manager._initialize_configuration = failing_config
        
        success = await app_manager.initialize_application()
        
        if not success:
            print(f"\n⚠️  Initialization failed as expected!")
            print(f"   State: {app_manager.get_application_state().value}")
            
            # Show error history
            progress_history = app_manager.get_initialization_progress()
            error_entries = [p for p in progress_history if p.error]
            
            if error_entries:
                print(f"   Error details: {error_entries[-1].error}")
        
    except Exception as e:
        print(f"❌ Demo failed with unexpected error: {e}")


async def demo_cleanup_tasks():
    """Demonstrate cleanup task registration and execution."""
    print("\n" + "="*60)
    print("🧹 DEMO: Cleanup Task Management")
    print("="*60)
    
    app_manager = ApplicationLifecycleManager()
    
    # Register some demo cleanup tasks
    cleanup_log = []
    
    def sync_cleanup_task():
        cleanup_log.append("Sync cleanup task executed")
        print("   🧹 Executed synchronous cleanup task")
    
    async def async_cleanup_task():
        cleanup_log.append("Async cleanup task executed")
        print("   🧹 Executed asynchronous cleanup task")
    
    def failing_cleanup_task():
        cleanup_log.append("Failing cleanup task attempted")
        print("   ⚠️  Cleanup task failed (this is expected)")
        raise Exception("Simulated cleanup failure")
    
    print("\n📋 Registering cleanup tasks...")
    app_manager.register_cleanup_task(sync_cleanup_task)
    app_manager.register_cleanup_task(async_cleanup_task)
    app_manager.register_cleanup_task(failing_cleanup_task)
    
    print("\n📋 Executing cleanup tasks...")
    await app_manager._execute_cleanup_tasks()
    
    print(f"\n✅ Cleanup completed!")
    print(f"   Tasks executed: {len(cleanup_log)}")
    for log_entry in cleanup_log:
        print(f"   - {log_entry}")


async def demo_signal_handling():
    """Demonstrate signal handling setup."""
    print("\n" + "="*60)
    print("📡 DEMO: Signal Handling")
    print("="*60)
    
    app_manager = ApplicationLifecycleManager()
    
    print("\n📋 Setting up interrupt handlers...")
    app_manager.setup_interrupt_handlers()
    
    print("✅ Signal handlers registered!")
    print("   - SIGINT (Ctrl+C) handler: ✓")
    print("   - SIGTERM handler: ✓")
    if hasattr(sys, 'platform') and sys.platform == 'win32':
        print("   - SIGBREAK handler: ✓")
    
    print(f"\n📋 Shutdown requested status: {app_manager.is_shutdown_requested()}")
    
    # Simulate setting shutdown flag
    app_manager._shutdown_requested = True
    print(f"   After simulation: {app_manager.is_shutdown_requested()}")


async def main():
    """Run all lifecycle demos."""
    print("🎭 Voice Character Transformation - Application Lifecycle Demo")
    print("=" * 70)
    
    # Setup logging
    setup_demo_logging()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Run demos
        await demo_successful_lifecycle()
        await demo_initialization_error()
        await demo_cleanup_tasks()
        await demo_signal_handling()
        
        print("\n" + "="*70)
        print("🎉 All lifecycle demos completed successfully!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())