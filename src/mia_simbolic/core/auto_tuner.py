"""
Auto-tuner module for M.I.A.-simbolic.

This module implements the BayesianAutoTuner class, which automatically
tunes hyperparameters for the optimization process using Bayesian optimization.
"""

import logging
import numpy as np
from typing import Callable, Dict, Any, Optional, List, Tuple, Union
import time

from ..config import Config

logger = logging.getLogger(__name__)

class BayesianAutoTuner:
    """Bayesian Auto-Tuner for hyperparameter optimization.
    
    This class uses Bayesian optimization to automatically tune
    hyperparameters for the optimization process.
    
    Attributes:
        config: Configuration object
        param_space: Parameter space for optimization
        best_params: Best parameters found
        n_trials: Number of trials for optimization
        n_init_points: Number of initial points for optimization
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Bayesian auto-tuner.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.n_trials = self.config.auto_tune_trials
        self.n_init_points = self.config.auto_tune_init_points
        
        # Define parameter space
        self.param_space = {
            'learning_rate': (1e-4, 1.0, 'log-uniform'),
            'momentum': (0.0, 0.99, 'uniform'),
            'update_frequency': (1, 20, 'int'),
            'validation_threshold': (0.8, 0.99, 'uniform'),
            'alpha': (0.1, 0.9, 'uniform'),
            'beta': (0.05, 0.5, 'uniform'),
            'gamma': (0.01, 0.3, 'uniform')
        }
        
        self.best_params = None
        
        logger.debug(f"Initialized BayesianAutoTuner with n_trials={self.n_trials}, n_init_points={self.n_init_points}")
    
    def tune(self, 
            objective_function: Callable[[np.ndarray], float], 
            initial_point: np.ndarray,
            bounds: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """Tune hyperparameters for the optimization process.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point for optimization
            bounds: Bounds for optimization variables
            
        Returns:
            Dictionary of best hyperparameters
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
        except ImportError:
            logger.warning("scikit-optimize not found, using default parameters")
            self.best_params = self._get_default_params()
            return self.best_params
        
        logger.info("Starting Bayesian hyperparameter optimization")
        start_time = time.time()
        
        # Define parameter space for skopt
        dimensions = []
        for param_name, (low, high, param_type) in self.param_space.items():
            if param_type == 'int':
                dimensions.append(Integer(low, high, name=param_name))
            elif param_type == 'uniform':
                dimensions.append(Real(low, high, prior='uniform', name=param_name))
            elif param_type == 'log-uniform':
                dimensions.append(Real(low, high, prior='log-uniform', name=param_name))
        
        # Define objective function for hyperparameter optimization
        @use_named_args(dimensions=dimensions)
        def hyperopt_objective(**params):
            # Create a simple optimizer with these parameters
            from .optimizer import MIAOptimizer
            
            # Override config with these parameters
            config_dict = vars(self.config).copy()
            config_dict.update(params)
            
            # Normalize multi-objective weights
            alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
            total = alpha + beta + gamma
            config_dict['alpha'] = alpha / total
            config_dict['beta'] = beta / total
            config_dict['gamma'] = gamma / total
            
            # Create temporary config
            temp_config = Config(**config_dict)
            
            # Create optimizer with limited iterations
            optimizer = MIAOptimizer(
                convergence_tolerance=temp_config.convergence_tolerance,
                max_iterations=min(50, temp_config.max_iterations // 5),  # Limit iterations for speed
                auto_tune=False  # Disable auto-tuning to avoid recursion
            )
            
            # Run optimization
            try:
                result = optimizer.optimize(objective_function, initial_point.copy(), bounds)
                
                # Compute score (lower is better)
                score = result.fun
                
                # Penalize non-convergence
                if not result.converged:
                    score += 100.0
                
                # Penalize slow convergence
                score += 0.1 * result.nit
                
                return score
            
            except Exception as e:
                logger.warning(f"Error during hyperparameter evaluation: {e}")
                return 1000.0  # High penalty for errors
        
        # Run Bayesian optimization
        try:
            result = gp_minimize(
                hyperopt_objective,
                dimensions,
                n_calls=self.n_trials,
                n_initial_points=self.n_init_points,
                random_state=42,
                verbose=True
            )
            
            # Extract best parameters
            self.best_params = {}
            for i, param_name in enumerate([dim.name for dim in dimensions]):
                self.best_params[param_name] = result.x[i]
            
            # Normalize multi-objective weights
            alpha, beta, gamma = self.best_params['alpha'], self.best_params['beta'], self.best_params['gamma']
            total = alpha + beta + gamma
            self.best_params['alpha'] = alpha / total
            self.best_params['beta'] = beta / total
            self.best_params['gamma'] = gamma / total
            
            end_time = time.time()
            logger.info(f"Hyperparameter optimization completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Best parameters: {self.best_params}")
            
            return self.best_params
        
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {e}")
            logger.warning("Using default parameters")
            self.best_params = self._get_default_params()
            return self.best_params
    
    def get_params(self) -> Dict[str, Any]:
        """Get the best parameters found.
        
        Returns:
            Dictionary of best parameters
        """
        if self.best_params is None:
            return self._get_default_params()
        
        return self.best_params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'update_frequency': 5,
            'validation_threshold': 0.95,
            'alpha': 0.6,
            'beta': 0.3,
            'gamma': 0.1
        }