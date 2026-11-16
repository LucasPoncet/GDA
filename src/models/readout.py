from torch import nn

class Readout(nn.Module):
    """Map the embedding of each graph's readout node to class logits"""
    
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, h, readout_mask):
        return self.lin(h[readout_mask])