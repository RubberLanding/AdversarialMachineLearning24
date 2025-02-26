import torch
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

import json
from torch.nn import CrossEntropyLoss, Conv2d
from torchvision.models import resnet18
from torch.optim import SGD, Adam
import torch.optim.lr_scheduler as lr_scheduler

from autoattack import AutoAttack

# Add the project root directory to the sys path to make import from src
import sys
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from src.util import generate_run_name, NormalizeModel, set_rand_seed
from src.losses import LossWrapper
from src.attacks import fgsm, pgd_linf, pgd_linf_trades
from src.train import train_epoch_adversarial, eval_epoch_adversarial, eval_epoch

import argparse
from datetime import datetime

###########################
# PARSE INPUT ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs",
    type=int,
    default=1,
)
args = parser.parse_args()

###########################
# LOAD THE DATA
current_file_dir = Path(__file__).parent
data_dir = current_file_dir.parents[1] / "datasets"

batch_size = 1024

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
    ])
set_rand_seed()
data_train = CIFAR10(root=data_dir, train=True, download=False, transform=transform)
data_test = CIFAR10(root=data_dir, train=False, download=False, transform=transform)
data_test, data_val = torch.utils.data.random_split(data_test, [0.1, 0.9])

dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(data_test, batch_size=len(data_test), shuffle=False)
dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)

num_classes = len(data_train.classes)

data_train_subset = Subset(data_train, list(range(2*batch_size)))
data_val_subset = Subset(data_val, list(range(batch_size)))
data_test_subset = Subset(data_test, list(range(10)))

dataloader_train_subset = DataLoader(data_train_subset, batch_size=batch_size, shuffle=True)
dataloader_val_subset = DataLoader(data_val_subset, batch_size=batch_size, shuffle=False)
dataloader_test_subset = DataLoader(data_test_subset, batch_size=len(data_test_subset), shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###########################
# LOAD THE MODEL
weight_dir = current_file_dir.parents[1] / "weights"
weight_file = weight_dir / "resnet18.pt"

model_adv = resnet18()
model_adv.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model_adv.fc = torch.nn.Linear(model_adv.fc.in_features, num_classes)

pretrained_weights = torch.load(weight_file, weights_only=True)
model_adv.load_state_dict(pretrained_weights)
model_adv = model_adv.to(device)

model_adv = NormalizeModel(model_adv, device)

# ###########################
# # SET LOGGING
results_dir = current_file_dir.parents[1] / "results"
run_dir = results_dir / generate_run_name()
run_dir.mkdir(parents=True, exist_ok=True)
log = {key: [] for key in ["train_losses", "test_losses", "adv_losses",
                           "train_errors", "test_errors", "adv_errors",
                           "autoattack_acc"]}


###########################
# SET TRAINING PARAMETERS
epochs = args.epochs

opt = Adam(model_adv.parameters(), lr=1e-3)
# opt = SGD(model_adv.parameters(), lr=1e-1)
scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

trades = LossWrapper("TRADES")
cross_entropy = LossWrapper("CE")


###########################
# START TRAINING
print(f"Begin training (ADV): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Run: {run_dir.stem}\n")
print(*("TR      ", "TE      ", "ADV     ", "Epoch   "), sep="\t")

for t in range(epochs):
    train_err, train_loss = train_epoch_adversarial(dataloader_train, model_adv, pgd_linf, opt, loss_fn=cross_entropy, device=device)
    test_err, test_loss = eval_epoch(dataloader_val, model_adv, loss_fn=cross_entropy, device=device)
    adv_err, adv_loss = eval_epoch_adversarial(dataloader_val, model_adv, fgsm, loss_fn=cross_entropy, device=device)

    # Update the losses and errors
    log["train_losses"] += [train_loss]
    log["test_losses"] += [test_loss]
    log["adv_losses"] += [adv_loss]
    log["train_errors"] += [train_err]
    log["test_errors"] += [test_err]
    log["adv_errors"] += [adv_err]

    print(*("{:.6f}".format(train_err),
            "{:.6f}".format(test_err),
            "{:.6f}".format(adv_err),
            f"{t+1}",), sep="\t", flush=True)

    if t % 5 == 0:
        with open(run_dir / "log.json", "w") as f:
            json.dump(log, f)
        torch.save(model_adv.state_dict(), run_dir / "model_adv.pt")

print(f"End training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


###########################
# APPLY AUTOATTACK
X, y = next(iter(dataloader_test))
X, y = X.to(device), y.to(device)

print(f"Begin Autoattack: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
log["autoattack_acc"] = [autoattack_acc]
print(f"End Autoattack: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


###########################
# STORE RESULTS
with open(run_dir / "log.json", "w") as f:
    json.dump(log, f)
torch.save(model_adv.state_dict(), run_dir / "model_adv.pt")
