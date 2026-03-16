import numpy as np


def patch_numpy_deprecated_aliases() -> None:
    """Patch removed NumPy aliases used by legacy dependencies."""
    if not hasattr(np, "int"):
        np.int = int
