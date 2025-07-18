"""
Unit tests for the Config class.
"""

import unittest
import os
import tempfile
import yaml
from src.mia_simbolic.config import Config, load_config


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        # Check default values
        self.assertEqual(config.convergence_tolerance, 1e-6)
        self.assertEqual(config.max_iterations, 1000)
        self.assertTrue(config.early_stopping)
        self.assertEqual(config.patience, 10)
        
        self.assertEqual(config.generator_learning_rate, 0.01)
        self.assertEqual(config.orchestrator_update_frequency, 5)
        self.assertEqual(config.validation_threshold, 0.95)
        
        self.assertTrue(config.auto_tune)
        self.assertEqual(config.auto_tune_trials, 20)
        self.assertEqual(config.auto_tune_init_points, 5)
        
        self.assertEqual(config.alpha, 0.6)
        self.assertEqual(config.beta, 0.3)
        self.assertEqual(config.gamma, 0.1)
        
        self.assertEqual(config.num_threads, 4)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.log_level, "INFO")
        
        self.assertEqual(config.validation_frequency, 10)
        self.assertEqual(config.validation_split, 0.2)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid log level
        config = Config(log_level="DEBUG")
        self.assertEqual(config.log_level, "DEBUG")
        
        # Test invalid log level
        with self.assertRaises(ValueError):
            Config(log_level="INVALID")
        
        # Test valid device
        config = Config(device="cuda")
        self.assertEqual(config.device, "cuda")
        
        # Test invalid device
        with self.assertRaises(ValueError):
            Config(device="gpu")

    def test_load_config(self):
        """Test loading configuration from file."""
        # Create a temporary config file
        config_dict = {
            'convergence_tolerance': 1e-5,
            'max_iterations': 500,
            'generator_learning_rate': 0.05,
            'alpha': 0.7,
            'beta': 0.2,
            'gamma': 0.1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            # Load config from file
            config = load_config(config_path)
            
            # Check loaded values
            self.assertEqual(config.convergence_tolerance, 1e-5)
            self.assertEqual(config.max_iterations, 500)
            self.assertEqual(config.generator_learning_rate, 0.05)
            self.assertEqual(config.alpha, 0.7)
            self.assertEqual(config.beta, 0.2)
            self.assertEqual(config.gamma, 0.1)
            
            # Check that other values are still default
            self.assertTrue(config.early_stopping)
            self.assertEqual(config.patience, 10)
            self.assertTrue(config.auto_tune)
            
        finally:
            # Clean up
            os.unlink(config_path)

    def test_load_nonexistent_config(self):
        """Test loading configuration from a nonexistent file."""
        config = load_config("nonexistent_file.yaml")
        
        # Should return default config
        self.assertEqual(config.convergence_tolerance, 1e-6)
        self.assertEqual(config.max_iterations, 1000)
        self.assertEqual(config.generator_learning_rate, 0.01)

    def test_load_invalid_config(self):
        """Test loading configuration from an invalid file."""
        # Create a temporary invalid config file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("invalid: yaml: file:")
            config_path = f.name
        
        try:
            # Load config from invalid file
            config = load_config(config_path)
            
            # Should return default config
            self.assertEqual(config.convergence_tolerance, 1e-6)
            self.assertEqual(config.max_iterations, 1000)
            self.assertEqual(config.generator_learning_rate, 0.01)
            
        finally:
            # Clean up
            os.unlink(config_path)


if __name__ == '__main__':
    unittest.main()