import torch

import torch.nn as nn
import torch.optim as optim

# Define a simple linear neural network
class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input, one output

    def forward(self, x):
        return self.linear(x)

# Example usage
if __name__ == "__main__":
    # Create dataset: y = 2x + 1
    x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

    model = LinearNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}")


    # Print weight and bias after training
    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    print(f"Trained weight: {weight:.2f}, bias: {bias:.2f}")
    
    # Test the model
    test_input = torch.tensor([[5.0]])
    predicted = model(test_input)
    print(f"Predicted output for input 5.0: {predicted.item():.2f}")