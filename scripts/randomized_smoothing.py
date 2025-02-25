import json
import torch
from torch.nn import CrossEntropyLoss, Conv2d
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from datetime import datetime

# Add the project root directory to the sys path to make import from src
import sys
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from src.randomized_smoothing import plot_certified_acc
from src.util import generate_run_name, NormalizeModel


###########################
# LOAD THE DATA
current_file_dir = Path(__file__).parent
data_dir = current_file_dir.parents[1] / "datasets"

batch_size = 1024
debug = True

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
    ])

data_test = CIFAR10(root=data_dir, train=False, download=False, transform=transform)
data_test, data_val = torch.utils.data.random_split(data_test, [0.1, 0.9])
dataloader_test = DataLoader(data_test, batch_size=len(data_test), shuffle=False)

data_test_subset = Subset(data_test, list(range(100)))
dataloader_test_subset = DataLoader(data_test_subset, batch_size=len(data_test_subset), shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###########################
# LOAD THE MODEL

current_file_dir = Path(__file__).parent


result_dir = current_file_dir.parents[1] / "results"
run_dir = result_dir / "white-bright-hawk-20250223-1413"
adv_weight_file = run_dir / "model_adv.pt"

weight_dir = current_file_dir.parents[1] / "weights"
weight_file = weight_dir / "resnet18.pt"

def get_resnet18():
  model = resnet18()
  model.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  model.fc = torch.nn.Linear(model.fc.in_features, 10)
  return model

model_base = get_resnet18()
pretrained_weights = torch.load(weight_file, weights_only=True)
model_base.load_state_dict(pretrained_weights)
model_base = NormalizeModel(model_base, device)
model_base = model_base.to(device)

model_adv = get_resnet18()
model_adv = NormalizeModel(model_adv, device)
pretrained_weights_adv = torch.load(adv_weight_file, weights_only=True, map_location=device)
model_adv.load_state_dict(pretrained_weights_adv)
model_adv = model_adv.to(device)

X, y = next(iter(dataloader_test_subset))
X, y = X.to(device), y.to(device)

print(f"Begin certification base model: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
plot_certified_acc(model_base, X, y, num_samples_estimation=7500,
                       num_samples_selection=100, plot_file=run_dir / "model_base")
print(f"End certification base model: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Begin certification adversarial model: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
plot_certified_acc(model_adv, X, y, num_samples_estimation=7500,
                       num_samples_selection=100, plot_file=run_dir / "model_adv")
print(f"End certification adversarial model: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
