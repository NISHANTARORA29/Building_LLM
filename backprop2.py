# Backpropagation: tiny worked example + gradient check
# (Binary classification, 1 hidden layer with tanh)

import numpy as np

np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(42)

# Data (single example)
x = rng.normal(size=(3,))           # input vector (3,)
y = np.array(1.0)                   # target scalar in {0,1}

# Parameters
W1 = rng.normal(scale=0.5, size=(2, 3))  # (hidden=2, in=3)
b1 = rng.normal(scale=0.5, size=(2,))    # (hidden,)
W2 = rng.normal(scale=0.5, size=(1, 2))  # (out=1, hidden=2)
b2 = rng.normal(scale=0.5, size=(1,))    # (out,)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(x, W1, b1, W2, b2, y):
    # layer 1
    z1 = W1 @ x + b1          # (2,)
    a1 = np.tanh(z1)          # (2,)
    # layer 2
    z2 = W2 @ a1 + b2         # (1,)
    y_hat = sigmoid(z2)       # (1,)
    # binary cross-entropy (scalar)
    eps = 1e-12
    L = - (y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))
    return L.item(), (x, z1, a1, z2, y_hat)

def backward(cache, W1, b1, W2, b2, y):
    x, z1, a1, z2, y_hat = cache
    # For sigmoid + BCE, dL/dz2 = y_hat - y
    dz2 = (y_hat - y).reshape(1,)             # (1,)
    dW2 = dz2.reshape(1,1) @ a1.reshape(1,2)  # (1,2)
    db2 = dz2                                  # (1,)
    da1 = W2.T @ dz2.reshape(1,)               # (2,)
    dz1 = da1 * (1 - a1**2)                    # tanh'(z1) = 1 - tanh^2
    dW1 = dz1.reshape(2,1) @ x.reshape(1,3)    # (2,3)
    db1 = dz1                                  # (2,)
    return dict(dW1=dW1, db1=db1, dW2=dW2, db2=db2)

# Compute analytic gradients
L, cache = forward(x, W1, b1, W2, b2, y)
grads = backward(cache, W1, b1, W2, b2, y)

# Finite-difference gradient checking (central difference)
def pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

def unpack(theta):
    i = 0
    W1_ = theta[i:i+2*3].reshape(2,3); i += 2*3
    b1_ = theta[i:i+2].reshape(2,);    i += 2
    W2_ = theta[i:i+1*2].reshape(1,2); i += 1*2
    b2_ = theta[i:i+1].reshape(1,);    i += 1
    return W1_, b1_, W2_, b2_

theta = pack(W1, b1, W2, b2)

# Flatten analytic grads in the same order
g_analytic = pack(grads['dW1'], grads['db1'], grads['dW2'], grads['db2'])

def loss_from_theta(theta):
    W1_, b1_, W2_, b2_ = unpack(theta)
    L_, _ = forward(x, W1_, b1_, W2_, b2_, y)
    return L_

eps = 1e-6
g_numeric = np.zeros_like(theta)
for i in range(theta.size):
    t_plus = theta.copy();  t_plus[i] += eps
    t_minus = theta.copy(); t_minus[i] -= eps
    Lp = loss_from_theta(t_plus)
    Lm = loss_from_theta(t_minus)
    g_numeric[i] = (Lp - Lm) / (2*eps)

# Compare
abs_diff = np.abs(g_numeric - g_analytic)
rel_diff = abs_diff / (np.maximum(1e-12, np.abs(g_numeric) + np.abs(g_analytic)))

print("Forward pass:")
print(f"  Loss L = {L:.6f}")
print("\nAnalytic gradients (shapes):")
print(f"  dW1 {grads['dW1'].shape}:\n{grads['dW1']}")
print(f"  db1 {grads['db1'].shape}:\n{grads['db1']}")
print(f"  dW2 {grads['dW2'].shape}:\n{grads['dW2']}")
print(f"  db2 {grads['db2'].shape}:\n{grads['db2']}")

print("\nGradient check:")
print(f"  max |g_num - g_an|   = {abs_diff.max():.3e}")
print(f"  mean |g_num - g_an|  = {abs_diff.mean():.3e}")
print(f"  max relative error   = {rel_diff.max():.3e}")
print("  (values < 1e-6 are typically considered an excellent match)")
