from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_ibrnet import IBRLLFFDataset 
from .dataset_llff_test import LLFFTestDataset 
from .waymo import WaymoStaticDataset
from .dataset_llff import LLFFDataset 
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "ibrnet": IBRLLFFDataset,
    "llff_test": LLFFTestDataset,
    "llff": LLFFDataset,
    "waymo": WaymoStaticDataset,
}


DatasetCfg = DatasetRE10kCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
