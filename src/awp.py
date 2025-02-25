import torch
import torch.nn as nn
import torch.nn.functional as F

class AWP(object):
    def __init__(self, model, optimizer, adv_param="weight", adv_lr=0.01, adv_eps=5e-3):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.grad_backup = {}

    def attack_backward(self, x, y):
        """
        This is the function to call to execute the attack.
        It will update the model weights in-place.
        """
        if self.adv_param == "weight":
            self._save()  # Save model weights before perturbation
            self._attack_step()  # Perform adversarial weight perturbation
        elif self.adv_param == "gradient":
            self._attack_step_grad()  # Perform adversarial gradient perturbation
        else:
            raise ValueError(f"Invalid adv_param: {self.adv_param}")

        # Calculate loss with perturbed weights
        loss = F.cross_entropy(self.model(x), y)

        # Update model parameters
        self.optimizer.zero_grad()  # Reset gradients before backward pass
        loss.backward()
        self.optimizer.step()  # Update model weights

        if self.adv_param == "weight":
            self._restore()  # Restore original model weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and norm2 != 0:
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - self.adv_eps),
                        self.backup[name] + self.adv_eps,
                    )

    def _attack_step_grad(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                # Compute the adversarial gradient perturbation
                # Here, it's a simple scaled version of the gradient
                r_at = self.adv_lr * param.grad

                # Add the perturbation to the gradient
                param.grad.add_(r_at)

                # Optional: Clipping the gradient
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.grad_backup[name] = param.grad.clone()

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])