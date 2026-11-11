# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from pathlib import Path

from torch.utils.data import Sampler

from ultralytics.utils import LOGGER


class SizeWeightedSampler(Sampler):
    """Repeat small-object images within each modality's original training quota."""

    def __init__(self, dataset, weight, seed, small_side=16):
        if not math.isfinite(weight) or weight < 1:
            raise ValueError("small_image_weight must be a finite number greater than or equal to 1")
        train_sources = dataset.data["train"]
        if not isinstance(train_sources, (list, tuple)) or len(train_sources) != 2:
            raise ValueError("Size-weighted sampling expects separate RGB and IR training directories")
        roots = [Path(source).resolve() for source in train_sources]
        if len(set(roots)) != len(roots) or not all(root.is_dir() for root in roots):
            raise ValueError("Size-weighted sampling expects two distinct training directories")

        self.length = len(dataset)
        self.random = random.Random(seed)
        groups = {root: {"background": [], "regular": [], "small": []} for root in roots}
        for index, label in enumerate(dataset.labels):
            image = Path(label["im_file"]).resolve()
            matches = [root for root in roots if image.is_relative_to(root)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one modality directory for {image}")
            boxes = label["bboxes"]
            if len(boxes):
                height, width = label["shape"]
                scale = dataset.imgsz / max(height, width)
                resized_h = min(math.ceil(height * scale), dataset.imgsz)
                resized_w = min(math.ceil(width * scale), dataset.imgsz)
                shortest = min((boxes[:, 2] * resized_w).min(), (boxes[:, 3] * resized_h).min())
                group = "small" if shortest < small_side else "regular"
            else:
                group = "background"
            groups[matches[0]][group].append(index)

        self.draws = []
        for root, group in groups.items():
            small = group["small"]
            regular = group["regular"]
            background = group["background"]
            positives = len(small) + len(regular)
            weighted_small = len(small) * weight
            small_count = round(positives * (weighted_small / (weighted_small + len(regular)))) if positives else 0
            self.draws.extend(((small, small_count), (regular, positives - small_count), (background, len(background))))
            LOGGER.info(
                f"Size sampling {root.parent.parent.name}: {positives + len(background)} draws/epoch, "
                f"{len(background)} background, {len(small)} -> {small_count} with a box shorter than {small_side}px"
            )

    def __iter__(self):
        indices = []
        for pool, count in self.draws:
            if count:
                repeats, extra = divmod(count, len(pool))
                indices.extend(pool * repeats)
                indices.extend(self.random.sample(pool, extra))
        self.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.length
