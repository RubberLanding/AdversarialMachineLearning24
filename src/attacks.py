import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def fgsm(model, X, y, epsilon=8/255):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=8/255, alpha=2/255, num_iter=10, randomize=True):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_trades(model, X, y, epsilon=8/255, alpha=2/255, num_iter=10, randomize=True):
    """ Construct FGSM adversarial examples with KL-Divergence for TRADES.
        Should be used together with traded_loss().
    """
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        # maybe set log_target=True and pass F.log_softmax(model(X), dim=1),
        # see the docs for more details
        loss = F.kl_div(F.log_softmax(model(X + delta), dim=1),
                        F.softmax(model(X), dim=1),
                        reduction='batchmean')
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
