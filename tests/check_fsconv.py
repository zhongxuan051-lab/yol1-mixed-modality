"""Check the full supplied FSConv architecture; synthetic data are not an AP benchmark."""
import argparse
import ast
import contextlib
import csv
import importlib.util
import io
import math
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
from pytorch_wavelets import DWTForward

from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.fsconv import FSConv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import yaml_load

MODEL = ROOT / "ultralytics/cfg/models/11/yolo11n-fsconv-full.yaml"
POSITIONS = [0, 1, 3, 5, 7, 17, 20]


def check_wavelet():
    """Independent 2x2 Haar sums/differences and derivatives, including odd sizes."""
    for h, w in ((8, 10), (9, 11)):
        x = torch.randn(2, 3, h, w, dtype=torch.double, requires_grad=True)
        padded = F.pad(x, (0, w % 2, 0, h % 2))
        a, b = padded[:, :, ::2, ::2], padded[:, :, ::2, 1::2]
        c, d = padded[:, :, 1::2, ::2], padded[:, :, 1::2, 1::2]
        expected = torch.stack(((a + b + c + d) / 2, (a + b - c - d) / 2,
                                (a - b + c - d) / 2, (a - b - c + d) / 2), dim=2)
        low, high = DWTForward(J=1, mode="zero", wave="haar").double()(x)
        actual = torch.cat((low.unsqueeze(2), high[0]), dim=2)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        probe = torch.randn_like(actual)
        grad = torch.autograd.grad((actual * probe).sum(), x)[0]
        ref = torch.autograd.grad((expected * probe).sum(), x)[0]
        torch.testing.assert_close(grad, ref, rtol=1e-6, atol=1e-6)
    print("PASS: independent Haar zero-padding/subband order/gradient checks")


def check_pack(pack):
    namespace = {"torch": torch, "nn": torch.nn, "init": torch.nn.init,
                 "math": math, "Conv": Conv, "DWTForward": DWTForward}
    for relative, name in (("attention.py", "SEAttention"), ("block.py", "FSConv")):
        path = pack / "ultralytics/nn/extra_modules" / relative
        source = path.read_text(encoding="utf-8-sig")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    for c1, c2, h, w in ((3, 16, 18, 22), (16, 32, 17, 19), (128, 256, 8, 10)):
        original = namespace["FSConv"](c1, c2).double().eval()
        port = FSConv(c1, c2).double().eval()
        port.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, c1, h, w, dtype=torch.double, requires_grad=True)
        reference = original(x)
        actual = port(x)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        probe = torch.randn_like(actual)
        original_grad = torch.autograd.grad((reference * probe).sum(), (x, *original.parameters()))
        port_grad = torch.autograd.grad((actual * probe).sum(), (x, *port.parameters()))
        for before, after in zip(original_grad, port_grad):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
    expected = yaml_load(pack / "ultralytics/cfg/models/11/yolo11-FSConv.yaml")
    actual = yaml_load(MODEL)
    expected["nc"] = 1
    assert expected == actual
    print("PASS: exact supplied-pack YAML, module outputs, input gradients and all parameter gradients")


def check_model(output, device):
    baseline_cfg = yaml_load(ROOT / "ultralytics/cfg/models/11/yolo11.yaml")
    candidate_cfg = yaml_load(MODEL)
    baseline_layers = baseline_cfg["backbone"] + baseline_cfg["head"]
    candidate_layers = candidate_cfg["backbone"] + candidate_cfg["head"]
    assert len(baseline_layers) == len(candidate_layers) == 24
    assert [i for i, (before, after) in enumerate(zip(baseline_layers, candidate_layers)) if before != after] == POSITIONS
    net = YOLO(str(MODEL)).model.to(device)
    assert [i for i, layer in enumerate(net.model) if isinstance(layer, FSConv)] == POSITIONS
    assert type(net.model[-1]) is Detect and net.model[-1].nc == 1
    assert net.stride.tolist() == [8., 16., 32.]
    expected_channels = [(3, 16), (16, 32), (64, 64), (128, 128), (128, 256), (64, 64), (128, 128)]
    for index, (c1, c2) in zip(POSITIONS, expected_channels):
        layer = net.model[index]
        assert layer.c1 == c1 and layer.conv5.conv.out_channels == c2
    print(f"PASS: seven replacement positions, channels, original Detect and strides; {sum(p.numel() for p in net.parameters())} parameters")
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
    se_gradients = []
    for index in POSITIONS:
        layer = net.model[index]
        for branch in (layer.conv1, layer.conv2, layer.conv3, layer.conv4, layer.conv5):
            assert branch.conv.weight.grad is not None and branch.conv.weight.grad.abs().max() > 0
        se_grad = layer.se.fc[0].weight.grad
        # A channel-reduction ReLU can legitimately be inactive for this batch.
        assert se_grad is not None and se_grad.isfinite().all()
        se_gradients.append(se_grad.abs().max().item())
    assert any(value > 0 for value in se_gradients)
    optimizer.step()
    print(f"PASS: {device} loss/backward/AdamW, seven spatial/wavelet paths and finite SE gradients: {items.tolist()}")
    net.eval()
    with torch.no_grad():
        for h, w in ((640, 640), (160, 224)):
            image = torch.rand(1, 3, h, w, device=device)
            predictions = net(image)[0]
            assert predictions.shape == (1, 5, sum((h // s) * (w // s) for s in (8, 16, 32)))
            assert predictions.isfinite().all()
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
    print("PASS: 640/rectangular inference, Conv/BN fusion, half checkpoint reload and FP16 inference")


def check_validation_table(output):
    """Check the four real validation calls and table format using synthetic metrics."""
    spec = importlib.util.spec_from_file_location("fsconv_validation", ROOT / "val.py")
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
    assert len(rows) == 9 and rows[0] == ["验证集", "指标", "基线", "FSConv", "变化（百分点）"]
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
    with tempfile.TemporaryDirectory(prefix="fsconv-check-", dir=ROOT.parent) as directory:
        output = Path(directory)
        check_wavelet()
        if args.reference_pack:
            check_pack(args.reference_pack)
        check_model(output, device)
        check_validation_table(output)
        if args.smoke_train:
            smoke_train(output, device)
    print("ALL FSCONV CHECKS PASSED")


if __name__ == "__main__":
    main()
