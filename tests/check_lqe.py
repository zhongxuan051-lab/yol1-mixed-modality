"""Focused numerical and integration checks; synthetic data are not an AP benchmark."""

import argparse
import contextlib
import csv
import importlib.util
import io
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT.parent / "yolo-settings"))

import numpy as np
from PIL import Image
import torch

from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules.lqe import LQE, Detect_LQE

MODEL = ROOT / "ultralytics/cfg/models/11/yolo11n-lqe.yaml"


def check_statistics():
    """Use separate edge slices as a reference, so a wrong reshape cannot pass."""
    lqe = LQE().double()
    corners = torch.randn(2, 64, 3, 5, dtype=torch.double, requires_grad=True)
    expected = []
    for edge in range(4):
        probabilities = corners[:, edge * 16:(edge + 1) * 16].softmax(dim=1)
        topk = probabilities.sort(dim=1, descending=True).values[:, :4]
        expected.extend((topk, topk.mean(dim=1, keepdim=True)))
    actual = lqe.distribution_statistics(corners)
    torch.testing.assert_close(actual, torch.cat(expected, dim=1))
    changed = corners.detach().clone()
    changed[:, :16] = 0
    torch.testing.assert_close(actual[:, 5:], lqe.distribution_statistics(changed)[:, 5:])
    probe = torch.randn_like(actual)
    grad = torch.autograd.grad((actual * probe).sum(), corners)[0]
    ref_grad = torch.autograd.grad((torch.cat(expected, dim=1) * probe).sum(), corners)[0]
    torch.testing.assert_close(grad, ref_grad)
    # Once learned, the quality correction must reach the correct edge's bins.
    nn = torch.nn
    nn.init.constant_(lqe.reg_conf[0].weight, .1)
    nn.init.constant_(lqe.reg_conf[0].bias, 1.)
    nn.init.constant_(lqe.reg_conf[-1].weight, .1)
    lqe(torch.zeros(2, 1, 3, 5, dtype=torch.double), corners).sum().backward()
    assert corners.grad.isfinite().all() and corners.grad.abs().max() > 0
    print("PASS: independent per-edge statistics, isolation, and gradient reference")


def check_model(output, device):
    baseline = YOLO(str(ROOT / "ultralytics/cfg/models/11/yolo11.yaml")).model.to(device)
    net = YOLO(str(MODEL)).model.to(device)
    missing = net.load_state_dict(baseline.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert len(missing.missing_keys) == 12 and all(".lqe." in key for key in missing.missing_keys)
    assert len(net.model) == len(baseline.model) == 24
    assert all(type(net.model[i]) is type(baseline.model[i]) for i in range(23))
    head = net.model[-1]
    assert isinstance(head, Detect_LQE) and len(head.lqe) == 3
    assert head.f == [16, 19, 22] and head.nc == 1
    assert net.stride.tolist() == [8., 16., 32.]
    added = sum(p.numel() for p in net.parameters()) - sum(p.numel() for p in baseline.parameters())
    assert added == 4227
    baseline.eval()
    net.eval()
    with torch.no_grad():
        for h, w in ((640, 640), (160, 224)):
            image = torch.rand(1, 3, h, w, device=device)
            before = baseline(image)[0]
            after = net(image)[0]
            torch.testing.assert_close(after, before, rtol=0, atol=0)
            assert after.shape == (1, 5, sum((h // s) * (w // s) for s in (8, 16, 32)))
    print(f"PASS: unchanged backbone/neck, zero-init baseline equivalence at 640 and rectangular input; +{added} parameters")
    del baseline

    net.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    net.train()
    batch = {
        "img": torch.rand(2, 3, 128, 128, device=device),
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor([[.4, .4, .15, .15], [.6, .5, .2, .2]], device=device),
    }
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, items = net(batch)
        assert loss.isfinite().all()
        loss.sum().backward()
        assert all(p.grad.isfinite().all() for p in net.parameters() if p.grad is not None)
        for lqe in head.lqe:
            assert lqe.reg_conf[-1].weight.grad.abs().max() > 0
            if step:
                assert lqe.reg_conf[0].weight.grad.abs().max() > 0
        optimizer.step()
    print(f"PASS: {device} detection loss/backward and AdamW; both layers learn at every scale: {items.tolist()}")

    net.eval()
    image = torch.rand(1, 3, 160, 224, device=device)
    with torch.no_grad():
        expected = net(image)[0]
        fused = deepcopy(net).fuse(verbose=False)
        torch.testing.assert_close(expected, fused(image)[0], rtol=2e-3, atol=2e-4)
        exported = deepcopy(net)
        exported.model[-1].export = True
        exported.model[-1].format = "onnx"
        torch.testing.assert_close(expected, exported(image), rtol=0, atol=0)
    checkpoint = output / "integration.pt"
    torch.save({"model": deepcopy(net).half().cpu(), "train_args": {"task": "detect"}}, checkpoint)
    loaded = YOLO(str(checkpoint)).model.to(device).float().eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, loaded(image)[0], rtol=.02, atol=.05)
        if device == "cuda":
            assert loaded.half()(image.half())[0].isfinite().all()
    print("PASS: Conv/BN fusion, export forward contract, FP16 checkpoint reload and inference")


def check_validation_table(output):
    """Check the four real validation calls and table format using synthetic metrics."""
    spec = importlib.util.spec_from_file_location("lqe_validation", ROOT / "val.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = output / "validation-fixture"
    data.mkdir()
    for name in ("data_rgb.yaml", "data.yaml", "baseline.pt", "candidate.pt"):
        (data / name).touch()
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = Path(weights)

        def val(self, **kwargs):
            calls.append((self.weights.name, kwargs))
            offset = .01 if self.weights.name == "candidate.pt" else 0
            return SimpleNamespace(box=SimpleNamespace(map50=.8 + offset, map75=.5 + offset, map=.4 + offset, mr=.7 - offset))

    table_dir = output / "validation-table"
    argv = ["val.py", "--dataset-root", str(data), "--baseline-weights", str(data / "baseline.pt"),
            "--weights", str(data / "candidate.pt"), "--output-dir", str(table_dir)]
    with patch.object(module, "YOLO", FakeYOLO), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        module.main()
    assert len(calls) == 4
    assert [(name, Path(kw["data"]).name) for name, kw in calls] == [
        ("baseline.pt", "data_rgb.yaml"), ("baseline.pt", "data.yaml"),
        ("candidate.pt", "data_rgb.yaml"), ("candidate.pt", "data.yaml"),
    ]
    assert all(all(kw[key] == value for key, value in {"split": "val", "imgsz": 640, "batch": 4, "workers": 0, "device": "1"}.items()) for _, kw in calls)
    with (table_dir / "comparison.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    assert len(rows) == 9 and rows[0] == ["验证集", "指标", "基线", "LQE", "变化（百分点）"]
    assert [row[1] for row in rows[1:5]] == ["mAP50", "AP75", "mAP50–95", "Recall"]
    assert rows[1][2:] == ["0.8000", "0.8100", "+1.00"] and rows[4][-1] == "-1.00"
    print("PASS: separate baseline/candidate RGB/IR val calls, four metrics and signed percentage-point table (mock metrics)")


def smoke_train(output, device):
    settings.update({"wandb": False})
    data = output / "smoke-data"
    rng = np.random.default_rng(0)
    for split, count in (("train", 4), ("val", 2)):
        for kind in ("images", "labels"):
            (data / kind / split).mkdir(parents=True)
        for i in range(count):
            array = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(array).save(data / "images" / split / f"{i}.jpg")
            (data / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.15 0.15\n", encoding="utf-8")
    yaml = data / "data.yaml"
    yaml.write_text(f"path: {data.as_posix()}\ntrain: images/train\nval: images/val\nnames: ['person']\n", encoding="utf-8")
    device_id = 0 if device == "cuda" else "cpu"
    YOLO(str(MODEL)).train(
        data=str(yaml), epochs=1, batch=2, imgsz=128, device=device_id,
        workers=0, amp=False, optimizer="AdamW", lr0=.001, seed=0,
        project=str(output / "smoke-runs"), name="integration", exist_ok=True,
        plots=False, verbose=False, save=True, val=True,
    )
    weights = output / "smoke-runs/integration/weights/best.pt"
    assert weights.is_file()
    YOLO(str(weights)).val(data=str(yaml), split="val", imgsz=128, batch=1, workers=0,
                          device=device_id, half=device == "cuda", plots=False,
                          project=str(output / "smoke-runs"), name="reload-val", exist_ok=True)
    print("PASS: synthetic full trainer, EMA, best.pt saving, reload and validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-train", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="lqe-check-", dir=ROOT.parent) as directory:
        output = Path(directory)
        check_statistics()
        check_model(output, device)
        check_validation_table(output)
        if args.smoke_train:
            smoke_train(output, device)
    print("ALL LQE CHECKS PASSED")


if __name__ == "__main__":
    main()
