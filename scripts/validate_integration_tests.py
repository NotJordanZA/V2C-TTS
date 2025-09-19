#!/usr/bin/env python3
"""
Validation script to verify integration test setup and basic functionality.

This script performs basic validation of the integration test environment
and runs a subset of tests to verify everything is working correctly.
"""

import sys
import os
import json
from pathlib import Path
import subprocess
import importlib.util


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Requires Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'pytest',
        'pytest-asyncio',
        'numpy',
        'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def check_project_structure():
    """Check if project structure is correct."""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "src",
        "tests", 
        "characters",
        "config"
    ]
    
    required_files = [
        "src/core/pipeline.py",
        "src/core/config.py",
        "tests/test_end_to_end_integration.py",
        "tests/test_performance_benchmarks.py",
        "characters/anime-waifu.json"
    ]
    
    missing_items = []
    
    for directory in required_dirs:
        dir_path = project_root / directory
        if not dir_path.exists():
            print(f"❌ Missing directory: {directory}")
            missing_items.append(directory)
        else:
            print(f"✅ Directory exists: {directory}")
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {file_path}")
            missing_items.append(file_path)
        else:
            print(f"✅ File exists: {file_path}")
    
    return len(missing_items) == 0


def validate_character_profiles():
    """Validate character profile files."""
    project_root = Path(__file__).parent.parent
    characters_dir = project_root / "characters"
    
    if not characters_dir.exists():
        print("❌ Characters directory not found")
        return False
    
    character_files = list(characters_dir.glob("*.json"))
    if len(character_files) == 0:
        print("❌ No character profile files found")
        return False
    
    required_fields = [
        "name", "description", "personality_traits", "speech_patterns",
        "vocabulary_preferences", "transformation_prompt", "voice_model_path"
    ]
    
    valid_profiles = 0
    
    for character_file in character_files:
        if character_file.name == ".gitkeep":
            continue
        
        try:
            with open(character_file, 'r') as f:
                character_data = json.load(f)
            
            missing_fields = []
            for field in required_fields:
                if field not in character_data:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ {character_file.name}: Missing fields {missing_fields}")
            else:
                print(f"✅ {character_file.name}: Valid character profile")
                valid_profiles += 1
                
        except json.JSONDecodeError as e:
            print(f"❌ {character_file.name}: Invalid JSON - {e}")
        except Exception as e:
            print(f"❌ {character_file.name}: Error reading file - {e}")
    
    return valid_profiles > 0


def run_basic_test():
    """Run a basic test to verify the test framework works."""
    project_root = Path(__file__).parent.parent
    
    # Try to run a simple test
    test_command = [
        sys.executable, "-m", "pytest",
        "tests/test_end_to_end_integration.py::TestAutomatedTestSuite::test_character_profile_validation",
        "-v", "--tb=short"
    ]
    
    try:
        result = subprocess.run(
            test_command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Basic test execution successful")
            return True
        else:
            print("❌ Basic test execution failed")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Basic test execution timed out")
        return False
    except Exception as e:
        print(f"❌ Error running basic test: {e}")
        return False


def create_test_directories():
    """Create necessary test directories."""
    project_root = Path(__file__).parent.parent
    
    directories = [
        "models/voices",
        "models/llm",
        "models/tts",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep file
        gitkeep_path = dir_path / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()
        
        print(f"✅ Created directory: {directory}")
    
    return True


def main():
    """Main validation function."""
    print("🔍 Validating Integration Test Environment")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Character Profiles", validate_character_profiles),
        ("Test Directories", create_test_directories),
        ("Basic Test Execution", run_basic_test)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_function in checks:
        print(f"\n📋 {check_name}")
        print("-" * 30)
        
        try:
            if check_function():
                passed_checks += 1
                print(f"✅ {check_name}: PASSED")
            else:
                print(f"❌ {check_name}: FAILED")
        except Exception as e:
            print(f"❌ {check_name}: ERROR - {e}")
    
    print(f"\n{'=' * 50}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Passed: {passed_checks}/{total_checks}")
    print(f"Success Rate: {passed_checks/total_checks*100:.1f}%")
    
    if passed_checks == total_checks:
        print("🎉 All validation checks passed! Integration tests are ready to run.")
        print("\nTo run integration tests:")
        print("  python scripts/run_integration_tests.py --test-type integration")
        print("\nTo run all tests:")
        print("  python scripts/run_integration_tests.py --test-type all")
        return True
    else:
        print("⚠️  Some validation checks failed. Please fix the issues before running integration tests.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)