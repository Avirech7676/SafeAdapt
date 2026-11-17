"""Training module for SafeAdapt-OT.

Provides training routines for ResNet-18 on Hospital 0 (Center 0 - RadboudUMC)
across 10,000, 20,000, and 30,000 sample allocations.
"""

from .train import train_source_model, main

__all__ = ["train_source_model", "main"]

