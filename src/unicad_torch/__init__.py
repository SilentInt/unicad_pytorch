from importlib.metadata import version as _version

from unicad_torch.callbacks import (
    Callback,
    CallbackManager,
    EarlyStopping,
    FitContext,
    History,
    LRScheduler,
    ModelCheckpoint,
    TqdmProgress,
)
from unicad_torch.config import GalaxyConfig
from unicad_torch.galaxy import Galaxy
from unicad_torch.smm_torch import SMMTorch

__version__ = _version("unicad-torch")
__all__ = [
    "Galaxy",
    "GalaxyConfig",
    "SMMTorch",
    "Callback",
    "CallbackManager",
    "EarlyStopping",
    "FitContext",
    "History",
    "LRScheduler",
    "ModelCheckpoint",
    "TqdmProgress",
]
