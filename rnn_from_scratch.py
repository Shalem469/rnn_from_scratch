import numpy as np

X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]]
], dtype=np.float32)

y = np.array([
    [4],
    [5],
    [6],
    [7],
    [8]
], dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)


# -------------------------
# RNN dimensions
# -------------------------

input_size = 1
hidden_size = 10
output_size = 1

print("Input size:", input_size)
print("Hidden size:", hidden_size)
print("Output size:", output_size)

# -------------------------
# Initialize weights
# -------------------------

np.random.seed(42)

Wx = np.random.randn(hidden_size, input_size) * 0.1
Wh = np.random.randn(hidden_size, hidden_size) * 0.1
b = np.zeros((hidden_size, 1))

Wy = np.random.randn(output_size, hidden_size) * 0.1
by = np.zeros((output_size, 1))

print("Wx:", Wx.shape)
print("Wh:", Wh.shape)
print("b :", b.shape)
print("Wy:", Wy.shape)
print("by:", by.shape)

h = np.zeros((hidden_size, 1))

print(h)
print("Shape:", h.shape)


# -------------------------
# Forward pass
# -------------------------
def forward(x):

    h = np.zeros((hidden_size, 1))

    hs = [h]

    for t in range(len(x)):

        xt = x[t].reshape(input_size, 1)

        h = np.tanh(
            Wx @ xt +
            Wh @ h +
            b
        )

        hs.append(h)

    y_pred = Wy @ h + by

    return y_pred, hs


prediction, hs = forward(X[0])

print("Prediction:")
print(prediction)

print("\nNumber of hidden states:")
print(len(hs))

target = y[0].reshape(1, 1)

error = prediction - target

loss = np.mean(error ** 2)

print("Prediction:", prediction)
print("Target:", target)
print("Loss:", loss)


# -------------------------
# Training
# -------------------------

epochs = 30000
learning_rate = 0.001
loss_history = []

for epoch in range(epochs):

    total_loss = 0

    for sample in range(len(X)):

        x = X[sample]
        target = y[sample].reshape(1, 1)

        # -------------------------
        # Forward pass
        # -------------------------

        prediction, hs = forward(x)

        # -------------------------
        # MSE Loss
        # -------------------------

        error = prediction - target
        loss = np.mean(error ** 2)

        total_loss += loss

        # -------------------------
        # Output gradients
        # -------------------------

        dy = 2 * error

        dWy = dy @ hs[-1].T
        dby = dy

        dh = Wy.T @ dy

        # -------------------------
        # RNN gradients
        # -------------------------

        dWx = np.zeros_like(Wx)
        dWh = np.zeros_like(Wh)
        db = np.zeros_like(b)

        # -------------------------
        # Backpropagation Through Time
        # -------------------------

        for t in reversed(range(len(x))):

            h = hs[t + 1]
            h_prev = hs[t]

            xt = x[t].reshape(input_size, 1)

            # Derivative of tanh
            dtanh = dh * (1 - h ** 2)

            dWx += dtanh @ xt.T
            dWh += dtanh @ h_prev.T
            db += dtanh

            dh = Wh.T @ dtanh

        # -------------------------
        # Gradient clipping
        # -------------------------

        np.clip(dWx, -1, 1, out=dWx)
        np.clip(dWh, -1, 1, out=dWh)
        np.clip(db, -1, 1, out=db)
        np.clip(dWy, -1, 1, out=dWy)
        np.clip(dby, -1, 1, out=dby)

        # -------------------------
        # Update parameters
        # -------------------------

        Wx -= learning_rate * dWx
        Wh -= learning_rate * dWh
        b -= learning_rate * db

        Wy -= learning_rate * dWy
        by -= learning_rate * dby

    # -------------------------
    # Print loss
    # -------------------------

    if epoch % 500 == 0:

        average_loss = total_loss / len(X)
        loss_history.append(average_loss)

        print(
            f"Epoch {epoch} | Loss: {average_loss:.4f}"
        )




for i in range(len(X)):

    prediction, _ = forward(X[i])

    print(
        f"Input: {X[i].flatten()} "
        f"Actual: {y[i][0]:.2f} "
        f"Prediction: {prediction[0][0]:.2f}"
    )


test=np.array(
    [[2], [3], [4]],
    dtype=np.float32
)
prediction,_=forward(test)
print(f"Prediction: {prediction[0][0]:.2f}")


import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Training Checkpoint")
plt.ylabel("Loss")
plt.title("RNN Training Loss")
plt.show()

