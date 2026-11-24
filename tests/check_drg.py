"""Independent dynamic-kernel and integration checks; synthetic data are not an AP benchmark."""
import argparse
import ast
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
import torch.nn.functional as F

from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules.block import C3k, C3k2
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.drg import DynamicSpatialAttention, RCSAB, DRG, C3k2_DRG
from ultralytics.utils import yaml_load

MODEL = ROOT / "ultralytics/cfg/models/11/yolo11n-drg-full.yaml"
POSITIONS = [2, 4, 6, 8, 13, 16, 19, 22]


def check_dynamic_attention():
    """Compare grouped computation to independent per-image convolution and gradients."""
    for batch_size in (1, 3):
        module = DynamicSpatialAttention(16).double()
        x = torch.randn(batch_size, 16, 5, 7, dtype=torch.double, requires_grad=True)
        actual = module(x)
        reference = []
        for sample in x.split(1):
            kernel = module.kernel_generator(sample).reshape(1, 1, 3, 3)
            weights = F.conv2d(sample.mean(dim=1, keepdim=True), kernel, padding=1).sigmoid()
            reference.append(sample * weights)
        expected = torch.cat(reference, dim=0)
        torch.testing.assert_close(actual, expected)
        probe = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual * probe).sum(), (x, *module.parameters()))
        reference_grads = torch.autograd.grad((expected * probe).sum(), (x, *module.parameters()))
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad)
        if batch_size > 1:
            changed = x.detach().clone()
            changed[0] *= 3
            torch.testing.assert_close(module(changed)[1:], actual[1:], rtol=0, atol=0)
    print("PASS: independent per-image kernel reference, all gradients and batch isolation (B=1/3)")


def check_reference(pack):
    path = pack / "ultralytics/nn/extra_modules/block.py"
    source = path.read_text(encoding="utf-8-sig")
    section = source.split("######################################## TGRS2025 DRPCA-Net start ########################################", 1)[1].split("######################################## TGRS2025 DRPCA-Net end ########################################", 1)[0]
    namespace = {"torch": torch, "nn": torch.nn, "F": F, "Conv": Conv, "C3k": C3k, "C3k2": C3k2}
    exec(compile(section, str(path), "exec"), namespace)
    for nested in (False, True):
        original = namespace["C3k2_DRG"](32, 64, c3k=nested).double()
        port = C3k2_DRG(32, 64, c3k=nested).double()
        port.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, 32, 7, 11, dtype=torch.double, requires_grad=True)
        for training in (False, True):
            original.train(training)
            port.train(training)
            expected = original(x)
            actual = port(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            probe = torch.randn_like(actual)
            ref_grad = torch.autograd.grad((expected * probe).sum(), (x, *original.parameters()))
            port_grad = torch.autograd.grad((actual * probe).sum(), (x, *port.parameters()))
            for before, after in zip(ref_grad, port_grad):
                torch.testing.assert_close(after, before, rtol=0, atol=0)
    expected_cfg = yaml_load(pack / "ultralytics/cfg/models/11/yolo11-C3k2-DRG.yaml")
    expected_cfg["nc"] = 1
    assert yaml_load(MODEL) == expected_cfg
    print("PASS: exact supplied-pack reference; ordinary/nested C3k2 train/eval outputs and all gradients; full YAML")


def check_model(output, device):
    baseline_cfg = yaml_load(ROOT / "ultralytics/cfg/models/11/yolo11.yaml")
    candidate_cfg = yaml_load(MODEL)
    baseline_layers = baseline_cfg["backbone"] + baseline_cfg["head"]
    candidate_layers = candidate_cfg["backbone"] + candidate_cfg["head"]
    assert len(baseline_layers) == len(candidate_layers) == 24
    assert [i for i, (before, after) in enumerate(zip(baseline_layers, candidate_layers)) if before != after] == POSITIONS
    net = YOLO(str(MODEL)).model.to(device)
    assert [i for i, layer in enumerate(net.model) if isinstance(layer, C3k2_DRG)] == POSITIONS
    assert type(net.model[-1]) is Detect and net.model[-1].nc == 1
    assert net.stride.tolist() == [8., 16., 32.]
    units = [m for m in net.modules() if isinstance(m, DRG)]
    assert len(units) == 11
    assert sum(isinstance(m, RCSAB) for m in net.modules()) == 11
    print(f"PASS: eight replacement positions, eleven DRG units, original Detect and strides; {sum(p.numel() for p in net.parameters())} parameters")
    net.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    net.train()
    batch = {
        "img": torch.rand(2, 3, 128, 128, device=device),
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor([[.4, .4, .15, .15], [.6, .5, .2, .2]], device=device),
    }
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    loss, items = net(batch)
    assert loss.isfinite().all()
    loss.sum().backward()
    assert all(p.grad.isfinite().all() for p in net.parameters() if p.grad is not None)
    for unit in units:
        assert all(p.grad is not None for p in unit.parameters())
        dynamic = next(m for m in unit.modules() if isinstance(m, DynamicSpatialAttention))
        assert dynamic.kernel_generator[-1].weight.grad.abs().max() > 0
    optimizer.step()
    print(f"PASS: {device} loss/backward/AdamW, all DRG gradients and eleven learned dynamic kernels: {items.tolist()}")
    net.eval()
    with torch.no_grad():
        for batch_size, h, w in ((1, 640, 640), (1, 160, 224), (4, 160, 224)):
            image = torch.rand(batch_size, 3, h, w, device=device)
            actual = net(image)[0]
            assert actual.shape == (batch_size, 5, sum((h // s) * (w // s) for s in (8, 16, 32)))
            assert actual.isfinite().all()
        expected = net(image)[0]
        fused = deepcopy(net).fuse(verbose=False)
        torch.testing.assert_close(expected, fused(image)[0], rtol=2e-3, atol=2e-4)
    checkpoint = output / "integration.pt"
    torch.save({"model": deepcopy(net).half().cpu(), "train_args": {"task": "detect"}}, checkpoint)
    loaded = YOLO(str(checkpoint)).model.to(device).float().eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, loaded(image)[0], rtol=.02, atol=.05)
        if device == "cuda":
            assert loaded.half()(image.half())[0].isfinite().all()
    print("PASS: 640/rectangular B=1/4 inference, Conv/BN fusion, half checkpoint reload and FP16 inference")


def check_validation_table(output):
    """Check the four real validation calls and table format using synthetic metrics."""
    spec = importlib.util.spec_from_file_location("drg_validation", ROOT / "val.py")
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
    assert len(rows) == 9 and rows[0] == ["验证集", "指标", "基线", "C3k2-DRG", "变化（百分点）"]
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
    parser.add_argument("--reference-pack", type=Path)
    parser.add_argument("--smoke-train", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="drg-check-", dir=ROOT.parent) as directory:
        output = Path(directory)
        check_dynamic_attention()
        if args.reference_pack:
            check_reference(args.reference_pack)
        check_model(output, device)
        check_validation_table(output)
        if args.smoke_train:
            smoke_train(output, device)
    print("ALL DRG CHECKS PASSED")


if __name__ == "__main__":
    main()
