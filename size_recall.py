"""Score-ordered, one-to-one IoU50 matching for recall at a false-positive cap."""

from itertools import groupby

import numpy as np

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import box_iou

SIZE_EDGES = (8, 16, 32)
SIZE_LABELS = ("<8", "8–<16", "16–<32", "≥32")


class SizeRecallValidator(DetectionValidator):
    """Collect size-recall inputs from the same predictions used in normal validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = np.zeros(len(SIZE_LABELS), dtype=int)
        self.detections = []

    def update_metrics(self, preds, batch):
        h, w = batch["img"].shape[-2:]
        for i, pred in enumerate(preds):
            labels = batch["bboxes"][batch["batch_idx"] == i]
            short_sides = (labels[:, 2:] * labels.new_tensor((w, h))).amin(dim=1)
            bins = np.searchsorted(SIZE_EDGES, short_sides.cpu().numpy(), side="right")
            self.targets += np.bincount(bins, minlength=len(SIZE_LABELS))
            if not len(pred):
                continue
            prepared = self._prepare_batch(i, batch)
            native = self._prepare_pred(pred, prepared)
            ious = box_iou(native[:, :4], prepared["bbox"]).cpu().numpy() if len(bins) else np.empty((len(pred), 0))
            scores = pred[:, 4].cpu().numpy()
            self.detections.extend(match_image(scores, ious, bins))
        super().update_metrics(preds, batch)


def match_image(scores, ious, bins):
    """Assign highest-IoU unmatched GT to each prediction in descending confidence."""
    claimed = np.zeros(len(bins), dtype=bool)
    for i in np.argsort(-scores, kind="stable"):
        size_bin = -1
        eligible = np.flatnonzero((ious[i] >= 0.5) & ~claimed)
        if eligible.size:
            target = eligible[np.argmax(ious[i, eligible])]
            claimed[target] = True
            size_bin = int(bins[target])
        yield float(scores[i]), size_bin


def recall_at_budget(detections, budget):
    """Return the deepest confidence cutoff within the budget, keeping equal scores together."""
    true_positives = np.zeros(len(SIZE_LABELS), dtype=int)
    false_positives = 0
    threshold = None
    ordered = sorted(detections, key=lambda record: -record[0])
    for score, group in groupby(ordered, key=lambda record: record[0]):
        matches = [size_bin for _, size_bin in group]
        additional_fp = matches.count(-1)
        if false_positives + additional_fp > budget:
            break
        false_positives += additional_fp
        for size_bin in matches:
            if size_bin >= 0:
                true_positives[size_bin] += 1
        threshold = score
    return true_positives, false_positives, threshold
