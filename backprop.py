import numpy as np

# Activation and derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(a):
    return (a > 0).astype(float)

# Loss and derivative (MSE)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.size

# Initialization
def initialize_parameters(input_dim, hidden_dim, output_dim):
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

# Forward pass
def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # for regression use identity, for classification use sigmoid
    return z1, a1, z2, a2

# Backward pass (gradient calculation using chain rule)
def backward(X, y, z1, a1, z2, a2, W2):
    m = X.shape[0]

    # Output layer gradients
    dz2 = mse_loss_deriv(y, a2) * sigmoid_deriv(a2)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Hidden layer gradients
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * sigmoid_deriv(a1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1/m, db1/m, dW2/m, db2/m

# Training loop
def train(X, y, input_dim, hidden_dim, output_dim, epochs=1000, lr=0.1):
    W1, b1, W2, b2 = initialize_parameters(input_dim, hidden_dim, output_dim)
    
    for epoch in range(epochs):
        # Forward
        z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)
        loss = mse_loss(y, a2)
        
        # Backward
        dW1, db1, dW2, db2 = backward(X, y, z1, a1, z2, a2, W2)
        
        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    return W1, b1, W2, b2

# Example data (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],])

trained_W1, trained_b1, trained_W2, trained_b2 = train(X, y, 2, 4, 1, epochs=2000, lr=0.2)

# Prediction
z1, a1, z2, preds = forward(X, trained_W1, trained_b1, trained_W2, trained_b2)
print("Predictions:\n", preds)
