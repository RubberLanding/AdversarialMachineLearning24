from pathlib import Path
import zipfile
import shutil

# First download the weigths into state_dicts.zip using "gdown https://drive.google.com/uc?id=17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq"
weight_dir = Path("../../../weights")
weight_file = weight_dir / "resnet18.pt"

zip_file = Path("path/to/zip")
"""Extract the pre-trained model weights to Google Drive"""
with zipfile.ZipFile(zip_file, "r") as zip_ref:
  with zip_ref.open("state_dicts/resnet18.pt") as zf, open(weight_file, 'wb') as f:
      shutil.copyfileobj(zf, f)
