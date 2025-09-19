#!/usr/bin/env python3
"""
Simple launcher script for the Voice Character Transformation application.

This script provides an easy way to run the application with proper
dependency checking and error handling.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    # Check core dependencies
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter (usually comes with Python)")
    
    # Optional dependencies
    optional_missing = []
    
    try:
        import loguru
    except ImportError:
        optional_missing.append("loguru")
    
    try:
        import psutil
    except ImportError:
        optional_missing.append("psutil")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with: pip install " + " ".join(missing_deps))
        return False
    
    if optional_missing:
        print("⚠️  Missing optional dependencies (app will still work):")
        for dep in optional_missing:
            print(f"   - {dep}")
        print("Install with: pip install " + " ".join(optional_missing))
    
    return True

def run_ui_demo():
    """Run the UI demo."""
    print("🎬 Starting UI Demo...")
    try:
        import ui_complete_demo
        return ui_complete_demo.main()
    except Exception as e:
        print(f"❌ UI Demo failed: {e}")
        return 1

def run_main_app():
    """Run the main application."""
    print("🚀 Starting Main Application...")
    try:
        from src.main import run_application
        run_application()
        return 0
    except Exception as e:
        print(f"❌ Main application failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def run_lifecycle_demo():
    """Run the lifecycle demo."""
    print("🎭 Starting Lifecycle Demo...")
    try:
        import examples.application_lifecycle_demo as demo
        import asyncio
        asyncio.run(demo.main())
        return 0
    except Exception as e:
        print(f"❌ Lifecycle demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def show_menu():
    """Show the application menu."""
    print("🎵 Voice Character Transformation System")
    print("=" * 50)
    print("Choose how to run the application:")
    print()
    print("1. 🎬 UI Demo (Recommended for first run)")
    print("2. 🚀 Main Application")
    print("3. 🎭 Lifecycle Demo")
    print("4. 🧪 Validate Environment")
    print("5. ❌ Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                return run_ui_demo()
            elif choice == "2":
                return run_main_app()
            elif choice == "3":
                return run_lifecycle_demo()
            elif choice == "4":
                return validate_environment()
            elif choice == "5":
                print("👋 Goodbye!")
                return 0
            else:
                print("❌ Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
        except EOFError:
            print("\n👋 Goodbye!")
            return 0

def validate_environment():
    """Validate the environment setup."""
    print("🔍 Validating Environment...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/validate_integration_tests.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

def main():
    """Main entry point."""
    print("🎵 Voice Character Transformation System Launcher")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot run application due to missing dependencies.")
        return 1
    
    print("✅ Dependencies check passed!")
    print()
    
    # Check if we have command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["ui", "demo"]:
            return run_ui_demo()
        elif arg in ["main", "app"]:
            return run_main_app()
        elif arg in ["lifecycle", "life"]:
            return run_lifecycle_demo()
        elif arg in ["validate", "check"]:
            return validate_environment()
        else:
            print(f"❌ Unknown argument: {arg}")
            print("Valid arguments: ui, main, lifecycle, validate")
            return 1
    
    # Show interactive menu
    return show_menu()

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Launcher failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)