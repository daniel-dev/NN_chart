import torch
import torch.nn as nn

# Create a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Create and save the model
model = SimpleNN()
# Save both the full model and just the state_dict for compatibility
torch.save(model, 'model_epoch10.pt')
torch.save(model.state_dict(), 'model_state_dict.pt')
print("Sample model saved as model_epoch10.pt")
print("Model state dict saved as model_state_dict.pt")
