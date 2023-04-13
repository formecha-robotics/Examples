import torch
import torch.nn as nn
import torch.optim as optim

class FullyConnectedSegmentation(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(FullyConnectedSegmentation, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Set the parameters for the fully connected neural network
input_size = 64 * 64 * 3  # Assuming RGB image, you can change it to 1 for grayscale
output_size = 64 * 64  # Binary segmentation mask (0 for background, 1 for person)
hidden_sizes = [1024, 512]

# Create the fully connected neural network
model = FullyConnectedSegmentation(input_size, output_size, hidden_sizes)

# Example usage:
# Load your image dataset and preprocess it to have images of size 64x64 and the corresponding segmentation masks
# Define your loss function (e.g., nn.BCELoss()) and optimizer (e.g., optim.Adam(model.parameters(), lr=0.001))

# Train the model on your dataset
