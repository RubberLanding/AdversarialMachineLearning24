import torch.nn.functional as F

class LossWrapper:
    def __init__(self, loss_fn, lambda_tradeoff=1.0):
        self.loss_fn = loss_fn
        self.lambda_tradeoff = lambda_tradeoff

    def __call__(self, model, X, y, delta=0.0):
        # Cross-Entropy Loss
        if self.loss_fn == "CE":
            yp = model(X + delta)
            return F.cross_entropy(yp, y)

        # TRADES Loss
        elif self.loss_fn == "TRADES":
            yp_adv = model(X + delta)
            yp_clean = model(X)
            clean_loss = F.cross_entropy(yp_clean, y)
            robust_loss = F.kl_div(F.log_softmax(yp_adv, dim=1),
                            F.softmax(yp_clean, dim=1),
                            reduction='batchmean')
            return clean_loss + self.lambda_tradeoff * robust_loss

        else:
            raise ValueError("Unsupported loss function")
