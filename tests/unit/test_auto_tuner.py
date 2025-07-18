"""
Unit tests for the BayesianAutoTuner class.
"""

import unittest
import numpy as np
from src.mia_simbolic.core.auto_tuner import BayesianAutoTuner
from src.mia_simbolic.config import Config


class TestBayesianAutoTuner(unittest.TestCase):
    """Test cases for the BayesianAutoTuner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.auto_tune_trials = 5  # Reduce for faster tests
        self.config.auto_tune_init_points = 2
        self.auto_tuner = BayesianAutoTuner(self.config)

    def test_initialization(self):
        """Test auto-tuner initialization."""
        self.assertEqual(self.auto_tuner.n_trials, self.config.auto_tune_trials)
        self.assertEqual(self.auto_tuner.n_init_points, self.config.auto_tune_init_points)
        self.assertIsNone(self.auto_tuner.best_params)
        
        # Check parameter space
        self.assertIn('learning_rate', self.auto_tuner.param_space)
        self.assertIn('momentum', self.auto_tuner.param_space)
        self.assertIn('update_frequency', self.auto_tuner.param_space)
        self.assertIn('validation_threshold', self.auto_tuner.param_space)
        self.assertIn('alpha', self.auto_tuner.param_space)
        self.assertIn('beta', self.auto_tuner.param_space)
        self.assertIn('gamma', self.auto_tuner.param_space)

    def test_get_default_params(self):
        """Test getting default parameters."""
        default_params = self.auto_tuner._get_default_params()
        
        self.assertIsInstance(default_params, dict)
        self.assertIn('learning_rate', default_params)
        self.assertIn('momentum', default_params)
        self.assertIn('update_frequency', default_params)
        self.assertIn('validation_threshold', default_params)
        self.assertIn('alpha', default_params)
        self.assertIn('beta', default_params)
        self.assertIn('gamma', default_params)
        
        # Check default values
        self.assertEqual(default_params['learning_rate'], 0.01)
        self.assertEqual(default_params['momentum'], 0.9)
        self.assertEqual(default_params['update_frequency'], 5)
        self.assertEqual(default_params['validation_threshold'], 0.95)
        self.assertEqual(default_params['alpha'], 0.6)
        self.assertEqual(default_params['beta'], 0.3)
        self.assertEqual(default_params['gamma'], 0.1)

    def test_get_params(self):
        """Test getting parameters."""
        # When best_params is None, should return default params
        params = self.auto_tuner.get_params()
        self.assertIsInstance(params, dict)
        self.assertEqual(params, self.auto_tuner._get_default_params())
        
        # Set best_params and check that get_params returns it
        test_params = {
            'learning_rate': 0.05,
            'momentum': 0.8,
            'update_frequency': 10,
            'validation_threshold': 0.9,
            'alpha': 0.5,
            'beta': 0.3,
            'gamma': 0.2
        }
        self.auto_tuner.best_params = test_params
        
        params = self.auto_tuner.get_params()
        self.assertEqual(params, test_params)

    def test_tune_simple_function(self):
        """Test tuning with a simple function."""
        def simple_function(x):
            return np.sum(x**2)
        
        initial_point = np.array([1.0, 1.0])
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        
        try:
            # This will use default params if scikit-optimize is not available
            params = self.auto_tuner.tune(simple_function, initial_point, bounds)
            
            self.assertIsInstance(params, dict)
            self.assertIn('learning_rate', params)
            self.assertIn('momentum', params)
            
            # Check that best_params is set
            self.assertEqual(self.auto_tuner.best_params, params)
            
        except ImportError:
            # If scikit-optimize is not available, tune should return default params
            params = self.auto_tuner.get_params()
            self.assertEqual(params, self.auto_tuner._get_default_params())


if __name__ == '__main__':
    unittest.main()