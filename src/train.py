import torch

def train_epoch(loader, model, opt, loss_fn, device):
    """Standard training/evaluation epoch over the dataset"""
    model.train()
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = loss_fn(model, X, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(loader, model, loss_fn, device):
    """Standard training/evaluation epoch over the dataset"""
    model.eval()
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = loss_fn(model, X, y)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def train_epoch_adversarial(loader, model, attack, opt, loss_fn, device, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    model.train()
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = loss_fn(model, X, y, delta)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def eval_epoch_adversarial(loader, model, attack, loss_fn, device, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    model.eval()  # Set the model to evaluation mode
    total_loss, total_err = 0.0, 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Compute adversarial perturbations (requires gradients)
        with torch.enable_grad():
            delta = attack(model, X, y, **kwargs)

        # Evaluate the model on adversarial examples without gradients
        with torch.no_grad():
            yp = model(X + delta)
            loss = loss_fn(model, X, y, delta)

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def train_epoch_awp(loader, model, attack, opt, loss_fn, awp, device, **kwargs):
    """Adversarial Weight Perturbation training/evaluation epoch over the dataset"""
    model.train()
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)

        #Perform AWP attack before forwad pass
        awp.attack_backward(X, y) # Apply perturbation

        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = loss_fn(model, X, y, delta)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)
