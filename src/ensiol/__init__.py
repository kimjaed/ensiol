"""EnsIOL: ensemble incremental online learning for transferable geophysical inversion."""
from .baselines import PCNConfig, PCNSampler
from .catalog import Catalog, MinDistanceCalibrator, Whitener
from .config import EnsIOLConfig, NoiseModel, ScoringConfig
from .context import AugmentationRun, ExperimentContext, station_seed
from .data import Grid, KilaueaDataset, Observation, PriorEnsemble, RealStation, SyntheticTestSet
from .enscgp import EnsCGP, Estimate
from .forward import MTForward1D, SerialExecutor
from .growth import EnsIOL, GrowthResult
from .metrics import Scorer, rms_log10
from .snapshots import SnapshotEvaluator

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
