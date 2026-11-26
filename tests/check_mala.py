"""Independent dense-attention reference and full single-stage MALA integration checks."""
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
from typing import Tuple
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
from ultralytics.nn.modules.block import C2PSA, PSABlock
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.mala import C2PSA_MALA, MALA, theta_shift
from ultralytics.utils import yaml_load

MODEL = ROOT / "ultralytics/cfg/models/11/yolo11n-mala.yaml"


def dense_reference(module, image):
    """Materialize token-to-token weights instead of using associative K^T V."""
    b, c, h, w = image.shape
    n = h * w
    qkvo = module.qkvo(image)
    q, k, v = [part.flatten(2).reshape(b, module.num_heads, module.head_dim, n).transpose(-2, -1)
               for part in qkvo[:, :3 * c].split(c, dim=1)]
    q, k = torch.nn.functional.elu(q) + 1, torch.nn.functional.elu(k) + 1
    z = (q * k.mean(dim=2, keepdim=True)).sum(dim=-1, keepdim=True) * module.scale
    sin, cos = module.repo((h, w))
    rotated = []
    for tensor in (q, k):
        pairs = tensor.reshape(*tensor.shape[:-1], -1, 2)
        perpendicular = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        rotated.append(tensor * cos + perpendicular * sin)
    attention = torch.einsum("bhid,bhjd->bhij", *rotated) * (module.scale / n)
    result = torch.einsum("bhij,bhjd->bhid", attention, v) * (1 + 1 / (z + 1e-6))
    result = result - z * v.mean(dim=2, keepdim=True)
    result = result.transpose(-2, -1).reshape(b, c, h, w)
    result = result + module.lepe(qkvo[:, 2 * c:3 * c])
    return module.proj(result * qkvo[:, 3 * c:])


def check_dense_reference():
    module = MALA(16, num_heads=2).double()
    image = torch.randn(2, 16, 3, 5, dtype=torch.double, requires_grad=True)
    actual, expected = module(image), dense_reference(module, image)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(), (image, *module.parameters()))
    expected_grads = torch.autograd.grad((expected * probe).sum(), (image, *module.parameters()))
    for grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, reference, rtol=1e-8, atol=1e-9)
    sin, cos = module.repo((3, 5))
    values = torch.randn(2, 2, 15, 8, dtype=torch.double)
    torch.testing.assert_close(theta_shift(values, sin, cos).square().sum(-1), values.square().sum(-1))
    zero = torch.zeros(1, 16, 3, 5, dtype=torch.double, requires_grad=True)
    module(zero).sum().backward()
    assert zero.grad.isfinite().all()
    print("PASS: independent dense token-matrix outputs/input/all-parameter gradients, rectangular 2D RoPE and zero-input backward")


def check_pack(pack):
    from einops import rearrange  # only used for optional comparison against the user's unmodified source
    path = pack / "ultralytics/nn/extra_modules/transformer.py"
    source = path.read_text(encoding="utf-8-sig")
    names = {"rotate_every_two", "theta_shift", "RoPE", "MALA", "PSABlock_MALA", "C2PSA_MALA"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    namespace = {"torch": torch, "nn": torch.nn, "Tuple": Tuple, "rearrange": rearrange,
                 "C2PSA": C2PSA, "PSABlock": PSABlock}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    original = namespace["C2PSA_MALA"](128, 128).double()
    port = C2PSA_MALA(128, 128).double()
    port.load_state_dict(original.state_dict(), strict=True)
    image = torch.randn(2, 128, 5, 7, dtype=torch.double, requires_grad=True)
    for training in (False, True):
        original.train(training)
        port.train(training)
        expected, actual = original(image), port(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        probe = torch.randn_like(actual)
        before = torch.autograd.grad((expected * probe).sum(), (image, *original.parameters()))
        after = torch.autograd.grad((actual * probe).sum(), (image, *port.parameters()))
        for ref_grad, actual_grad in zip(before, after):
            torch.testing.assert_close(actual_grad, ref_grad, rtol=0, atol=0)
    expected_cfg = yaml_load(pack / "ultralytics/cfg/models/11/yolo11-C2PSA-MALA.yaml")
    expected_cfg["nc"] = 1
    assert yaml_load(MODEL) == expected_cfg
    print("PASS: unchanged pack math/einops reference, full C2PSA train/eval outputs and all gradients; full YAML")


def check_model(output, device):
    baseline_cfg = yaml_load(ROOT / "ultralytics/cfg/models/11/yolo11.yaml")
    candidate_cfg = yaml_load(MODEL)
    before = baseline_cfg["backbone"] + baseline_cfg["head"]
    after = candidate_cfg["backbone"] + candidate_cfg["head"]
    assert len(before) == len(after) == 24
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [10]
    net = YOLO(str(MODEL)).model.to(device)
    assert [i for i, layer in enumerate(net.model) if isinstance(layer, C2PSA_MALA)] == [10]
    assert type(net.model[-1]) is Detect and net.model[-1].nc == 1
    assert net.stride.tolist() == [8., 16., 32.]
    block = net.model[10]
    assert block.c == 128 and len(block.m) == 1
    attention = block.m[0].attn
    assert isinstance(attention, MALA) and attention.dim == 128 and attention.num_heads == 2
    original_block = C2PSA(256, 256)
    for name in ("cv1", "cv2"):
        assert {key: value.shape for key, value in getattr(block, name).state_dict().items()} == {
            key: value.shape for key, value in getattr(original_block, name).state_dict().items()}
    assert {key: value.shape for key, value in block.m[0].ffn.state_dict().items()} == {
        key: value.shape for key, value in original_block.m[0].ffn.state_dict().items()}
    print(f"PASS: only layer 10 replaced; original C2PSA projections/FFN, Detect, strides; {sum(p.numel() for p in net.parameters())} parameters")
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
    for layer in (attention.qkvo, attention.lepe, attention.proj):
        assert layer.weight.grad is not None and layer.weight.grad.abs().max() > 0
    optimizer.step()
    print(f"PASS: {device} detection loss/backward/AdamW, learned QKVO/local-conv/projection gradients: {items.tolist()}")
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
    spec = importlib.util.spec_from_file_location("mala_validation", ROOT / "val.py")
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
    assert len(rows) == 9 and rows[0] == ["验证集", "指标", "基线", "C2PSA-MALA", "变化（百分点）"]
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
    with tempfile.TemporaryDirectory(prefix="mala-check-", dir=ROOT.parent) as directory:
        output = Path(directory)
        check_dense_reference()
        if args.reference_pack:
            check_pack(args.reference_pack)
        check_model(output, device)
        check_validation_table(output)
        if args.smoke_train:
            smoke_train(output, device)
    print("ALL MALA CHECKS PASSED")


if __name__ == "__main__":
    main()
