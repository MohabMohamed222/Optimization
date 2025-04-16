# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000, optimizer='gradient_descent', batch_size=32):
        # Initialize model parameters
        self.lr = learning_rate          # How fast the model learns
        self.n_iters = n_iters           # Number of training iterations
        self.optimizer = optimizer       # Optimization method to use
        self.batch_size = batch_size     # Size of data batches for mini-batch GD
        self.weights = None              # Model coefficients
        self.bias = None                # Model intercept
        self.loss_history = []          # To track loss during training

    def _sigmoid(self, z):
        # Convert inputs to probabilities between 0 and 1
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        # Calculate error using cross-entropy loss
        epsilon = 1e-15  # Prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        # Train the model using specified optimization method
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Start with zero weights
        self.bias = 0                       # Start with zero bias

        # Choose optimization method
        if self.optimizer == 'gradient_descent':
            self._gradient_descent(X, y)
        elif self.optimizer == 'batch_gradient_descent':
            self._batch_gradient_descent(X, y)
        elif self.optimizer == 'stochastic_gradient_descent':
            self._stochastic_gradient_descent(X, y)
        elif self.optimizer == 'mini_batch_gradient_descent':
            self._mini_batch_gradient_descent(X, y)
        else:
            raise ValueError("Unknown optimizer")

    # Below are different optimization methods ---------------------------------
    
    def _gradient_descent(self, X, y):
        # Standard gradient descent: Update using all samples at once
        for _ in range(self.n_iters):
            # Make predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Calculate and store loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Calculate gradients (slopes)
            dw = (1 / len(y)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(y)) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def _batch_gradient_descent(self, X, y):
        # Same as standard gradient descent
        self._gradient_descent(X, y)

    def _stochastic_gradient_descent(self, X, y):
        # Update weights using one sample at a time
        for _ in range(self.n_iters):
            for i in range(len(y)):
                # Pick random sample
                idx = np.random.randint(len(y))
                xi = X[idx:idx+1]
                yi = y[idx:idx+1]

                # Make prediction for single sample
                linear_model = np.dot(xi, self.weights) + self.bias
                y_pred = self._sigmoid(linear_model)

                # Calculate gradients
                dw = np.dot(xi.T, (y_pred - yi))
                db = np.sum(y_pred - yi)

                # Update weights and bias
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # Track loss after each epoch
            y_pred_all = self._sigmoid(np.dot(X, self.weights) + self.bias)
            epoch_loss = self._compute_loss(y, y_pred_all)
            self.loss_history.append(epoch_loss)

    def _mini_batch_gradient_descent(self, X, y):
        # Update weights using small data batches
        n_samples = len(y)

        for _ in range(self.n_iters):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Process in batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # Make predictions for batch
                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(linear_model)

                # Calculate gradients
                dw = (1 / len(y_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / len(y_batch)) * np.sum(y_pred - y_batch)

                # Update weights and bias
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # Track loss after each epoch
            y_pred_all = self._sigmoid(np.dot(X, self.weights) + self.bias)
            epoch_loss = self._compute_loss(y, y_pred_all)
            self.loss_history.append(epoch_loss)

    def predict(self, X, threshold=0.5):
        # Convert probabilities to class predictions (0 or 1)
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred >= threshold).astype(int)

    def evaluate(self, X, y):
        # Calculate performance metrics
        y_pred = self.predict(X)

        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

        # Calculate various metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }

        # Print results
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

        return metrics

# Create synthetic dataset for testing
X, y = make_classification(
    n_samples=1000,      # 1000 total samples
    n_features=10,       # 10 features per sample
    n_classes=2,         # Binary classification
    n_clusters_per_class=1,
    random_state=42      # For reproducibility
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data (scale to similar ranges)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Compare different optimization methods
optimizers = [
    ('gradient_descent', 'Gradient Descent'),
    ('batch_gradient_descent', 'Batch GD'),
    ('stochastic_gradient_descent', 'Stochastic GD'),
    ('mini_batch_gradient_descent', 'Mini-Batch GD')
]

# Create plot window
plt.figure(figsize=(15, 10))

# Train and compare all optimizers
for idx, (opt_name, display_name) in enumerate(optimizers, 1):
  
    # Create model with current optimizer
    model = LogisticRegression(
        learning_rate=0.1 
        if opt_name == 'stochastic_gradient_descent' else 0.01,
        n_iters=100,
        optimizer=opt_name,
        batch_size=64
    )

    # Train model
    model.fit(X_train, y_train)

    # Plot loss curve
    plt.subplot(2, 2, idx)
    plt.plot(model.loss_history)
    plt.title(f'{display_name} Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    # Print performance metrics
    print(f"\n{display_name} Evaluation:")
    model.evaluate(X_test, y_test)

# Show all plots
plt.tight_layout()
plt.show()
