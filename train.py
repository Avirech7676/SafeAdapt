"""SafeAdapt ResNet-18 Source Domain Training Entrypoint.

Trains ResNet-18 on Hospital 0 (RadboudUMC) across 10k, 20k, or 30k sample scales.
Delegates cleanly to the modular implementation in training_model.train.
"""

import sys
from pathlib import Path

# Ensure SafeAdapt_Fresh root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_model.train import main

if __name__ == "__main__":
    main()
