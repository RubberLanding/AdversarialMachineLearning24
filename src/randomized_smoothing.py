import torch
import pickle
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm, binomtest

class RandomizedSmoother(object):

  def __init__(self, base_model, num_classes, sigma=0.25, epsilon=8/255, alpha=0.05, norm="L2"):
    self.base_model = base_model
    self.num_classes = num_classes
    self.sigma = sigma
    self.epsilon = epsilon
    self.norm = norm
    self.alpha = alpha

  def predict(self, X, num_samples=100):
    batch_size = X.size(0)
    # Compute class counts for noisy samples
    class_counts = self.get_noisy_counts(X, num_samples)

    # Sort class counts to find top 2 most frequent classes
    sorted_counts_vals, sorted_counts_ind = class_counts.sort()
    top1_class = sorted_counts_ind[:, -1]  # Most frequent class
    top2_class = sorted_counts_ind[:, -2]  # Second most frequent class
    top1_class_count = sorted_counts_vals[:, -1]
    top2_class_count = sorted_counts_vals[:, -2]

    predictions = torch.zeros(batch_size, device=X.device)
    for i in range(batch_size):
        # Use binomial test to check if the top class is significantly more frequent
        n_a, n_b = top1_class_count[i], top2_class_count[i]
        predictions[i] = top1_class[i] if binomtest(n_a, n_a + n_b, p=0.5).pvalue <= self.alpha else float('nan')

    return predictions

  def certify(self, X, num_samples_selection=100, num_samples_estimation=100):
    batch_size = X.size(0)
    # Perform two sampling procedures to avoid selection bias
    counts_a = self.get_noisy_counts(X, num_samples_selection)
    top1_class = counts_a.argmax(dim=-1).unsqueeze(1)  # Get most frequent class index

    counts = self.get_noisy_counts(X, num_samples_estimation)
    # Extract the counts for the most frequent class
    n_a = counts.gather(dim=1, index=top1_class).squeeze()

    certified_radii = torch.zeros(batch_size)
    predictions = torch.zeros(batch_size, device=X.device)
    for i in range(batch_size):
        # Compute lower confidence bound for the top class probability
        conf_bound, _ = proportion_confint(n_a[i].cpu(), num_samples_estimation, alpha=2 * self.alpha, method="beta")
        # Calculate the certified radius based on the confidence bound
        if conf_bound >= 0.5:
          certified_radii[i] = self.get_radius(conf_bound, self.norm)
          predictions[i] = top1_class[i]
        else:
          certified_radii[i] = float("nan")
          predictions[i] = float("nan")

    return certified_radii, predictions

  def get_noisy_counts(self, X, num_samples):
    batch_size = X.size(0)
    class_counts = torch.zeros(batch_size, self.num_classes, dtype=torch.int64, device=X.device)

    for _ in range(num_samples):
        # Generate noisy samples based on the specified norm
        noisy_samples = X + self.get_noise(X, self.norm)
        with torch.no_grad():
            # Predict classes for noisy samples
            logits = self.base_model(noisy_samples)
            class_pred = F.softmax(logits, dim=-1).argmax(dim=-1)
            # Update class counts
            for i in range(batch_size):
                class_counts[i, class_pred[i]] += 1

    return class_counts

  def get_noise(self, X, p_norm="L2"):
    if p_norm=="L2":
      return torch.randn_like(X, device=X.device) * self.sigma
    elif p_norm=="Linf":
      return 2 * self.epsilon * torch.randn_like(X, device=X.device) - self.epsilon
    else:
      raise ValueError("Unsupported norm.")

  def get_radius(self, conf_bound, p_norm="L2"):
    if p_norm=="L2":
      return self.sigma * norm.ppf(conf_bound)
    elif p_norm=="Linf":
      return self.epsilon * (2 * conf_bound - 1)
    else:
      raise ValueError("Unsupported norm.")


def calculate_certified_acc(radius, yp, y, num_bins=1000):
  num_samples = len(y)
  ind_not_nan = ~torch.isnan(radius)
  radius, yp, y = radius[ind_not_nan], yp[ind_not_nan], y[ind_not_nan] # Filter out entries with NaN
  max_val = radius.max()
  bins = torch.linspace(0.0, max_val, num_bins + 1)
  counts = np.array([(yp[radius > bin_value] == y[radius > bin_value]).sum().cpu() for bin_value in bins])
  acc = counts / num_samples

  return bins, acc

def plot_certified_acc(model_base, X, y, num_classes=10, epsilon=8/255,
                       alpha=0.05, num_samples_estimation=2000,
                       num_samples_selection=100, plot_file=None):
  sigmas = [0.1, 0.25, 0.5]

  n_sigma = len(sigmas)
  cmap = plt.get_cmap("viridis", n_sigma)
  colors = [cmap(i) for i in range(n_sigma)]

  for i, sigma in enumerate(sigmas):
    color = colors[i]
    model_rs = RandomizedSmoother(model_base, num_classes, sigma, epsilon, alpha)
    radius, yp = model_rs.certify(X, num_samples_selection, num_samples_estimation)

    if torch.isnan(radius).all():
        print(f"Warning: All radii are NaN for sigma={sigma}. Skipping plotting.")
        continue  # Skip plotting if all radii are NaN
    x_radius, y_acc = calculate_certified_acc(radius, yp, y)
    plt.plot(x_radius, y_acc, label=f"sigma={sigma}", color=color, linewidth=1)
  plt.legend()

  if plot_file:
    plt.savefig(f"{plot_file}.png")

    fig = plt.gcf()
    with open(f"{plot_file}.pkl", "wb") as f:
        pickle.dump(fig, f)
    plt.close(fig)

  # plt.show()
