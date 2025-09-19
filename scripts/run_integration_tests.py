#!/usr/bin/env python3
"""
Integration test runner script for local development and CI/CD.

This script provides a convenient way to run different types of tests
with appropriate configurations and reporting.
"""

import argparse
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any


class TestRunner:
    """Test runner for integration and performance tests."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
    
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a command and capture results."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running: {description}")
            print(f"Command: {' '.join(command)}")
            print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=not self.verbose,
                text=True,
                check=False
            )
            
            success = result.returncode == 0
            
            if not success and not self.verbose:
                print(f"FAILED: {description}")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
            
            self.test_results[description] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout if not self.verbose else '',
                'stderr': result.stderr if not self.verbose else ''
            }
            
            return success
            
        except Exception as e:
            print(f"ERROR running {description}: {e}")
            self.test_results[description] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def setup_test_environment(self) -> bool:
        """Set up the test environment."""
        print("Setting up test environment...")
        
        # Create necessary directories
        directories = [
            "models/voices",
            "models/llm", 
            "models/tts",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files
            gitkeep_path = dir_path / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
        
        # Verify character profiles exist
        characters_dir = self.project_root / "characters"
        if not characters_dir.exists():
            print("ERROR: Characters directory not found")
            return False
        
        character_files = list(characters_dir.glob("*.json"))
        if len(character_files) == 0:
            print("ERROR: No character profile files found")
            return False
        
        print(f"Found {len(character_files)} character profiles")
        return True
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "-m", "not integration and not benchmark and not slow"
        ]
        
        if self.verbose:
            command.extend(["--capture=no"])
        
        return self.run_command(command, "Unit Tests")
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_end_to_end_integration.py",
            "-v",
            "--tb=short",
            "--integration"
        ]
        
        if self.verbose:
            command.extend(["--capture=no"])
        
        return self.run_command(command, "Integration Tests")
    
    def run_performance_tests(self) -> bool:
        """Run performance benchmark tests."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_performance_benchmarks.py",
            "-v",
            "--tb=short",
            "--benchmark"
        ]
        
        if self.verbose:
            command.extend(["--capture=no"])
        
        return self.run_command(command, "Performance Benchmarks")
    
    def run_configuration_tests(self) -> bool:
        """Run configuration validation tests."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_end_to_end_integration.py::TestAutomatedTestSuite",
            "-v",
            "--tb=short"
        ]
        
        if self.verbose:
            command.extend(["--capture=no"])
        
        return self.run_command(command, "Configuration Validation")
    
    def run_coverage_analysis(self) -> bool:
        """Run tests with coverage analysis."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "-m", "not benchmark"
        ]
        
        return self.run_command(command, "Coverage Analysis")
    
    def generate_report(self) -> None:
        """Generate a test report."""
        print(f"\n{'='*80}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - successful_tests
        
        print(f"Total test suites: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print(f"{'='*80}")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {test_name}")
            
            if not result['success'] and 'error' in result:
                print(f"    Error: {result['error']}")
            elif not result['success'] and result.get('stderr'):
                print(f"    Error output: {result['stderr'][:200]}...")
        
        # Save detailed results to file
        results_file = self.project_root / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'summary': {
                    'total': total_tests,
                    'successful': successful_tests,
                    'failed': failed_tests,
                    'success_rate': successful_tests/total_tests if total_tests > 0 else 0
                },
                'results': self.test_results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run integration tests for voice character transformation")
    
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "performance", "config", "coverage", "all"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip test environment setup"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    # Setup test environment
    if not args.no_setup:
        if not runner.setup_test_environment():
            print("Failed to set up test environment")
            sys.exit(1)
    
    # Run selected tests
    success = True
    
    if args.test_type in ["unit", "all"]:
        success &= runner.run_unit_tests()
    
    if args.test_type in ["integration", "all"]:
        success &= runner.run_integration_tests()
    
    if args.test_type in ["config", "all"]:
        success &= runner.run_configuration_tests()
    
    if args.test_type in ["coverage", "all"]:
        success &= runner.run_coverage_analysis()
    
    if args.test_type == "performance":
        success &= runner.run_performance_tests()
    
    # Generate report
    runner.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()