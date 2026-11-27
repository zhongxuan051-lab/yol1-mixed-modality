"""Pack equivalence and C2PSA-DyT detector/trainer integration checks."""
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

from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules.block import C2PSA, PSABlock, Attention
from ultralytics.nn.modules.dyt import DynamicTanh, PSABlock_DYT, C2PSA_DYT
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import yaml_load

MODEL = ROOT / "ultralytics/cfg/models/11/yolo11n-dyt.yaml"


def check_pack(pack):
    path = pack / "ultralytics/nn/extra_modules/transformer.py"
    source = path.read_text(encoding="utf-8-sig")
    names = {"DynamicTanh", "PSABlock_DYT", "C2PSA_DYT"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in names]
    port_nodes = [n for n in ast.parse((ROOT / "ultralytics/nn/modules/dyt.py").read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef)]
    assert [ast.dump(n, include_attributes=False) for n in nodes] == [ast.dump(n, include_attributes=False) for n in port_nodes]
    namespace = {"torch": torch, "nn": torch.nn, "C2PSA": C2PSA, "PSABlock": PSABlock}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    original, port = namespace["C2PSA_DYT"](256, 256).double(), C2PSA_DYT(256, 256).double()
    port.load_state_dict(original.state_dict(), strict=True)
    for training in (False, True):
        original.train(training)
        port.train(training)
        image = torch.randn(2, 256, 5, 7, dtype=torch.double, requires_grad=True)
        expected, actual = original(image), port(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        probe = torch.randn_like(actual)
        before = torch.autograd.grad((expected * probe).sum(), (image, *original.parameters()))
        after = torch.autograd.grad((actual * probe).sum(), (image, *port.parameters()))
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    expected = yaml_load(pack / "ultralytics/cfg/models/11/yolo11-C2PSA-DYT.yaml")
    expected["nc"] = 1
    assert yaml_load(MODEL) == expected
    print("PASS: all three class ASTs and full YAML match supplied pack; C2PSA-DyT train/eval outputs and all gradients identical")


def check_model(output, device):
    original_cfg = yaml_load(ROOT / "ultralytics/cfg/models/11/yolo11.yaml")
    candidate_cfg = yaml_load(MODEL)
    before, after = original_cfg["backbone"] + original_cfg["head"], candidate_cfg["backbone"] + candidate_cfg["head"]
    assert len(before) == len(after) == 24
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [10]
    net = YOLO(str(MODEL)).model.to(device)
    assert [i for i, m in enumerate(net.model) if isinstance(m, C2PSA_DYT)] == [10]
    assert type(net.model[-1]) is Detect and net.model[-1].nc == 1
    assert net.stride.tolist() == [8., 16., 32.]
    block = net.model[10]
    assert block.c == 128 and len(block.m) == 1
    assert type(block.m[0]) is PSABlock_DYT and type(block.m[0].attn) is Attention
    assert sum(isinstance(m, DynamicTanh) for m in net.modules()) == 2
    for module in (block.m[0].dyt1, block.m[0].dyt2):
        assert module.normalized_shape == 128 and module.channels_last is False
        torch.testing.assert_close(module.alpha, torch.tensor([.5], device=device), rtol=0, atol=0)
        assert torch.all(module.weight == 1) and torch.all(module.bias == 0)
    # Strip only the additional DyT parameters and verify every original layer remains.
    original = C2PSA(256, 256)
    retained = {name: value for name, value in block.state_dict().items() if ".dyt1." not in name and ".dyt2." not in name}
    original.load_state_dict(retained, strict=True)
    assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in block.modules()) == sum(isinstance(m, torch.nn.BatchNorm2d) for m in original.modules())
    print(f"PASS: only layer 10 changed; two 128-channel DyT operations, original Attention/FFN/BN/residuals and Detect/strides; {sum(p.numel() for p in net.parameters())} parameters")
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
    for layer in (block.m[0].dyt1, block.m[0].dyt2):
        for parameter in (layer.alpha, layer.weight, layer.bias):
            assert parameter.grad is not None and parameter.grad.abs().max() > 0
    optimizer.step()
    print(f"PASS: {device} loss/backward/AdamW and both DyT alpha/scale/bias gradients: {items.tolist()}")
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
            loaded.half()
            for batch_size, h, w in ((1, 640, 640), (4, 160, 224)):
                assert loaded(torch.rand(batch_size, 3, h, w, device=device).half())[0].isfinite().all()
    print("PASS: 640/rectangular B=1/4 inference, Conv/BN fusion, half checkpoint reload and FP16 inference")


def check_validation_table(output):
    """Check the four real validation calls and table format using synthetic metrics."""
    spec = importlib.util.spec_from_file_location("dyt_validation", ROOT / "val.py")
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
    assert len(rows) == 9 and rows[0] == ["验证集", "指标", "基线", "C2PSA-DyT", "变化（百分点）"]
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
    with tempfile.TemporaryDirectory(prefix="dyt-check-", dir=ROOT.parent) as directory:
        output = Path(directory)
        if args.reference_pack:
            check_pack(args.reference_pack)
        check_model(output, device)
        check_validation_table(output)
        if args.smoke_train:
            smoke_train(output, device)
    print("ALL C2PSA-DyT CHECKS PASSED")


if __name__ == "__main__":
    main()
