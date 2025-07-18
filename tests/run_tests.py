#!/usr/bin/env python3
"""
Test runner for M.I.A.-simbolic.

This script runs all tests for the M.I.A.-simbolic package.
"""

import os
import sys
import unittest
import argparse


def run_tests(test_type="all", verbose=False):
    """Run tests of the specified type.
    
    Args:
        test_type: Type of tests to run ("unit", "integration", "performance", or "all")
        verbose: Whether to print verbose output
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the path
    sys.path.insert(0, os.path.dirname(script_dir))
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Determine which tests to run
    if test_type == "unit" or test_type == "all":
        print("Running unit tests...")
        unit_tests = loader.discover(os.path.join(script_dir, "unit"))
        unit_result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(unit_tests)
        
        if unit_result.failures or unit_result.errors:
            print("Unit tests failed.")
            if test_type == "unit":
                return False
    
    if test_type == "integration" or test_type == "all":
        print("\nRunning integration tests...")
        integration_tests = loader.discover(os.path.join(script_dir, "integration"))
        integration_result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(integration_tests)
        
        if integration_result.failures or integration_result.errors:
            print("Integration tests failed.")
            if test_type == "integration":
                return False
    
    if test_type == "performance" or test_type == "all":
        print("\nRunning performance tests...")
        performance_tests = loader.discover(os.path.join(script_dir, "performance"))
        performance_result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(performance_tests)
        
        if performance_result.failures or performance_result.errors:
            print("Performance tests failed.")
            if test_type == "performance":
                return False
    
    if test_type == "all":
        return not (unit_result.failures or unit_result.errors or 
                   integration_result.failures or integration_result.errors or 
                   performance_result.failures or performance_result.errors)
    
    return True


def main():
    """Run the test runner."""
    parser = argparse.ArgumentParser(description="Run tests for M.I.A.-simbolic")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "all"],
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    success = run_tests(args.type, args.verbose)
    
    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nTests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()