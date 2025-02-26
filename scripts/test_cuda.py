import torch
import os

print("Number of GPUs detected:", torch.cuda.device_count())

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

print("CUDA available?", torch.cuda.is_available())
