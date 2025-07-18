"""
PyTorch integration example using M.I.A.-simbolic.

This example demonstrates how to use the MIAPyTorchOptimizer
with a simple neural network.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.mia_simbolic.integrations.pytorch import MIAPyTorchOptimizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install PyTorch to run this example.")


class SimpleNN(nn.Module):
    """Simple neural network for regression."""
    
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1):
        """Initialize the neural network."""
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_data(n_samples=100):
    """Generate synthetic data for regression."""
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.randn(n_samples, 1)
    return X, y


def main():
    """Run the PyTorch integration example."""
    if not TORCH_AVAILABLE:
        return
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data
    X, y = generate_data(n_samples=100)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create model
    model = SimpleNN(input_dim=1, hidden_dim=20, output_dim=1)
    
    # Create optimizer
    optimizer = MIAPyTorchOptimizer(
        model.parameters(),
        lr=0.01,
        auto_tune=False
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    n_epochs = 100
    losses = []
    
    print("Training neural network with MIAPyTorchOptimizer...")
    
    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X, y, color='blue', label='Training data')
    
    # Plot predictions
    plt.plot(X, y_pred, color='red', linewidth=2, label='Predictions')
    
    plt.title('Neural Network Regression with MIAPyTorchOptimizer')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('results/pytorch_integration_results.png')
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Save plot
    plt.savefig('results/pytorch_integration_loss.png')
    
    print("\nTraining completed.")
    print(f"Final loss: {losses[-1]:.6f}")
    print("Results saved to 'results' directory.")


if __name__ == "__main__":
    main()