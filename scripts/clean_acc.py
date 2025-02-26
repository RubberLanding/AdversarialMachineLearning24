import json
import torch
from torch.nn import CrossEntropyLoss, Conv2d
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from autoattack import AutoAttack
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from pathlib import Path

from datetime import datetime

# Add the project root directory to the sys path to make import from src
import sys
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from src.randomized_smoothing import plot_certified_acc
from src.util import generate_run_name, NormalizeModel, set_rand_seed


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

set_rand_seed()
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
run_id = "silver-bright-octopus-20250224-0606"
run_dir = result_dir / run_id

possible_model_names = ["model_ema.pt", "model_trades.pt", "model_awp.pt", "model_adv.pt"]
model_file = next((name for name in possible_model_names if (run_dir / name).exists()), None)
adv_weight_file = run_dir / model_file

def get_resnet18():
  model = resnet18()
  model.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  model.fc = torch.nn.Linear(model.fc.in_features, 10)
  return model

model_adv = get_resnet18()

if model_file == "model_ema.pt":
    model_adv = AveragedModel(model_adv, multi_avg_fn=get_ema_multi_avg_fn(0.9))

model_adv = NormalizeModel(model_adv, device)
pretrained_weights_adv = torch.load(adv_weight_file, weights_only=True, map_location=device)
model_adv.load_state_dict(pretrained_weights_adv)
model_adv = model_adv.to(device)

print(f"Begin Autoattack: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"Model Type: {model_file}")
print(f"Run ID: {run_id}\n")

X, y = next(iter(dataloader_test))
X, y = X.to(device), y.to(device)

with torch.no_grad():
    model_adv.eval()
    predictions = model_adv(X).argmax(dim=1)

adversary = AutoAttack(model_adv, norm='Linf', eps=8/255, version='standard', device=device)
adv_samples = adversary.run_standard_evaluation(X, y, bs=batch_size)

with torch.no_grad():
    model_adv.eval()
    adv_predictions = model_adv(adv_samples).argmax(dim=1)

autoattack_acc = (y == adv_predictions).float().mean().item() # AutoAttack reports this accuracy
# autoattack_acc = (predictions == adv_predictions).float().mean().item()
print(f"End Autoattack: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

#########################################################################################

# weight_dir = current_file_dir.parents[1] / "weights"
# weight_file = weight_dir / "resnet18.pt"
# model_adv = resnet18()
# model_adv.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# model_adv.fc = torch.nn.Linear(model_adv.fc.in_features, 10)
# pretrained_weights = torch.load(weight_file, weights_only=True)
# model_adv.load_state_dict(pretrained_weights)
# model_adv = model_adv.to(device)
# model_adv = NormalizeModel(model_adv, device)

# X, y = next(iter(dataloader_test))
# X, y = X.to(device), y.to(device)

# print(f"Number of samples in test set: {len(data_test)}")
# print(f"Shape of X: {X.shape}")

# print(f"Begin Evaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# print(f"Model Type: {model_file}")
# print(f"Run ID: {run_id}\n")

# with torch.no_grad():
#     model_adv.eval()
#     predictions = model_adv(X).argmax(dim=1)
# clean_acc = (y == predictions).float().mean().item()
# print(f"Clean Accuracy: {clean_acc:.3f}\n")
# print(f"End Evaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
