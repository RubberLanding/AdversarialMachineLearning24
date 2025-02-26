import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# Define a function to generate unique names for different model training runs
def generate_run_name():
    """Generate a random name for a run."""
    colors = [
        "red", "blue", "green", "yellow", "purple", "orange", "pink",
        "black", "white", "gray", "silver", "gold", "cyan", "magenta"]
    adjectives = [
        "fast", "slow", "shiny", "dull", "bright", "dark", "silent",
        "loud", "brave", "calm", "wise", "fierce", "kind", "strong"]
    nouns = [
        "dragon", "tiger", "lion", "panda", "wolf", "phoenix", "eagle",
        "fox", "bear", "shark", "hawk", "cheetah", "whale", "octopus"]
    color = random.choice(colors)
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{color}-{adjective}-{noun}-{timestamp}"
    return run_name

# Wrap model with normalization
class NormalizeModel(torch.nn.Module):
    def __init__(self, model, device, mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, x):
        return self.model((x - self.mean) / self.std)

def set_rand_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
