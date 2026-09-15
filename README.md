# RNN From Scratch

A vanilla Recurrent Neural Network (RNN) implemented from scratch using Python and NumPy.

The goal of this project is to understand how a Recurrent Neural Network works internally without using high-level deep learning frameworks such as TensorFlow or PyTorch.

## Project Overview

This project trains a simple RNN to learn a sequential numerical pattern.

The model receives a sequence of three numbers and predicts the next number.

### Training Examples

| Input Sequence | Target |
|---|---:|
| [1, 2, 3] | 4 |
| [2, 3, 4] | 5 |
| [3, 4, 5] | 6 |
| [4, 5, 6] | 7 |
| [5, 6, 7] | 8 |

## RNN Architecture

The RNN processes the input one timestep at a time while maintaining a hidden state.

```text
x₁ → RNN → h₁
           ↓
x₂ → RNN → h₂
           ↓
x₃ → RNN → h₃
           ↓
        Output
           ↓
      Prediction
```

## Mathematical Formulation

At each timestep, the hidden state is calculated as:

```text
hₜ = tanh(Wx × xₜ + Wh × hₜ₋₁ + b)
```

Where:

- `xₜ` is the input at timestep `t`
- `hₜ` is the current hidden state
- `hₜ₋₁` is the previous hidden state
- `Wx` is the input-to-hidden weight matrix
- `Wh` is the hidden-to-hidden weight matrix
- `b` is the hidden-layer bias

The final prediction is calculated as:

```text
ŷ = Wy × hₜ + by
```

Where:

- `Wy` is the hidden-to-output weight matrix
- `by` is the output bias
- `ŷ` is the predicted value

## Loss Function

The model uses Mean Squared Error (MSE) as the loss function:

```text
Loss = (ŷ - y)²
```

The gradient of the loss with respect to the prediction is:

```text
dL/dŷ = 2(ŷ - y)
```

## Implementation

The RNN is implemented using NumPy without using TensorFlow or PyTorch.

The implementation includes:

- RNN weight initialization
- Hidden-state computation
- Forward propagation
- Tanh activation
- Mean Squared Error (MSE)
- Output gradients
- Backpropagation Through Time (BPTT)
- Gradient clipping
- Gradient descent
- Sequence prediction
- Training-loss visualization

## Forward Pass

For each timestep, the RNN performs the following operations:

```text
Input xₜ
   ↓
Previous Hidden State hₜ₋₁
   ↓
Wx × xₜ + Wh × hₜ₋₁ + b
   ↓
tanh Activation
   ↓
New Hidden State hₜ
```

After processing the complete sequence, the final hidden state is used to generate the prediction.

## Backpropagation Through Time

The model uses Backpropagation Through Time (BPTT) to calculate gradients through the sequence.

During the backward pass, the model moves from the final timestep toward the first timestep.

```text
Forward:

x₁ → h₁ → x₂ → h₂ → x₃ → h₃ → Prediction


Backward:

x₁ ← h₁ ← x₂ ← h₂ ← x₃ ← h₃ ← Prediction
```

The gradients are calculated for:

```text
Wx
Wh
b
Wy
by
```

Gradient clipping is applied before updating the parameters to prevent excessively large gradients.

## Training Process

```text
Forward Pass
      ↓
Calculate Prediction
      ↓
Calculate MSE Loss
      ↓
Backpropagation Through Time
      ↓
Calculate Gradients
      ↓
Gradient Clipping
      ↓
Update Parameters
      ↓
Repeat
```

## Training Loss

The model records the training loss during training and visualizes the loss using Matplotlib.

![RNN Training Loss](training_loss.png)

The loss decreases rapidly because the dataset contains a very simple and predictable numerical pattern.

## Example Prediction

After training, the model can predict the next value in the sequence.

```text
Input:
[2, 3, 4]

Expected:
5

Prediction:
approximately 5
```

## Important Note

This project uses a very small synthetic dataset containing only five training examples.

Therefore, the purpose of this project is **not** to demonstrate real-world generalization.

Instead, the project is designed to demonstrate the internal mechanics of a vanilla RNN, including forward propagation, hidden states, BPTT, gradient calculation, and parameter updates.

## Technologies

- Python
- NumPy
- Matplotlib
- Git
- GitHub

## Concepts Learned

This project helped me understand:

- Recurrent Neural Networks
- Sequential data
- Hidden states
- Recurrent connections
- Tanh activation
- Forward propagation
- Mean Squared Error
- Backpropagation Through Time
- Gradient calculation
- Gradient clipping
- Gradient descent
- Sequence-to-one prediction

## Project Structure

```text
RNN_From_Scratch/
│
├── rnn_from_scratch.py
├── README.md
├── requirements.txt
├── .gitignore
└── training_loss.png
```

## Future Improvements

- Implement LSTM from scratch
- Implement GRU from scratch
- Use separate training, validation, and test datasets
- Experiment with longer sequences
- Implement sequence-to-sequence prediction
- Build an English-to-Spanish translation model
- Implement Attention from scratch
- Implement a Transformer from scratch

## Author

**Shalem469**

This project is part of my journey toward understanding Deep Learning from the fundamentals and building neural networks from scratch.