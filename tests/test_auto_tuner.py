"""
Test module for the auto-tuner functionality of M.I.A.-simbolic.

This module contains tests for the Bayesian auto-tuner component.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mia_simbolic import Config
from src.mia_simbolic.core.auto_tuner import BayesianAutoTuner


class TestBayesianAutoTuner(unittest.TestCase):
    """Test cases for the BayesianAutoTuner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create config with minimal settings for testing
        self.config = Config(
            auto_tune_trials=5,  # Reduced for faster testing
            auto_tune_init_points=2
        )
        
        # Create auto-tuner
        self.auto_tuner = BayesianAutoTuner(self.config)
        
        # Define a simple objective function for testing
        def sphere(x):
            return np.sum(x**2)
        
        self.objective = sphere
        self.initial_point = np.ones(5)
    
    def test_initialization(self):
        """Test initialization of the auto-tuner."""
        # Check if parameter space is defined
        self.assertIsInstance(self.auto_tuner.param_space, dict)
        self.assertGreater(len(self.auto_tuner.param_space), 0)
        
        # Check if trials and init points are set correctly
        self.assertEqual(self.auto_tuner.n_trials, self.config.auto_tune_trials)
        self.assertEqual(self.auto_tuner.n_init_points, self.config.auto_tune_init_points)
    
    def test_default_params(self):
        """Test default parameters."""
        # Get default parameters
        params = self.auto_tuner._get_default_params()
        
        # Check if all required parameters are present
        self.assertIn('learning_rate', params)
        self.assertIn('momentum', params)
        self.assertIn('update_frequency', params)
        self.assertIn('validation_threshold', params)
        self.assertIn('alpha', params)
        self.assertIn('beta', params)
        self.assertIn('gamma', params)
        
        # Check if weights sum to 1
        self.assertAlmostEqual(params['alpha'] + params['beta'] + params['gamma'], 1.0)
    
    def test_get_params(self):
        """Test getting parameters."""
        # Initially, best_params should be None
        self.assertIsNone(self.auto_tuner.best_params)
        
        # get_params should return default parameters
        params = self.auto_tuner.get_params()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
    
    def test_tune(self):
        """Test tuning hyperparameters."""
        try:
            from skopt import gp_minimize
            
            # Run tuning
            params = self.auto_tuner.tune(
                self.objective,
                self.initial_point
            )
            
            # Check if parameters were tuned
            self.assertIsNotNone(self.auto_tuner.best_params)
            self.assertIsInstance(params, dict)
            self.assertGreater(len(params), 0)
            
            # Check if weights sum to 1
            self.assertAlmostEqual(params['alpha'] + params['beta'] + params['gamma'], 1.0)
            
        except ImportError:
            # Skip test if scikit-optimize is not available
            self.skipTest("scikit-optimize not available")
    
    def test_fallback_to_defaults(self):
        """Test fallback to defaults when tuning fails."""
        # Create a failing objective function
        def failing_objective(x):
            raise ValueError("Simulated failure")
        
        # Run tuning with failing objective
        params = self.auto_tuner.tune(
            failing_objective,
            self.initial_point
        )
        
        # Should fall back to default parameters
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        
        # Check if weights sum to 1
        self.assertAlmostEqual(params['alpha'] + params['beta'] + params['gamma'], 1.0)


if __name__ == '__main__':
    unittest.main()