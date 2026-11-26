"""Check pack layout, bilinear math, compiled CUDA gradients and trainer integration."""
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
from torchvision.ops import deform_conv2d

from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.dcnv4 import C3k2_DCNv4, DCNv4, DCNV4_YOLO
from ultralytics.nn.modules.dcnv4_backend import DCNv4Function, dcnv4_reference, require_cuda_extension
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import yaml_load

MODEL = ROOT / "ultralytics/cfg/models/11/yolo11n-dcnv4.yaml"
ARGS = (3, 3, 1, 1, 1, 1, 1, 1, 1, 32, 1., 256, 0)


def torchvision_reference(value, packed, groups):
    """Independent deformable-convolution implementation of the 3x3 core."""
    batch, height, width, channels = value.shape
    packed = packed[..., :groups * 27].reshape(batch, height, width, groups, 27)
    # Pack kernel order is (x,y); torchvision enumerates (y,x), offsets (dy,dx).
    offset = packed[..., :18].reshape(batch, height, width, groups, 3, 3, 2)
    offset = offset.permute(0, 3, 5, 4, 6, 1, 2).flip(4).reshape(batch, groups * 18, height, width)
    mask = packed[..., 18:].reshape(batch, height, width, groups, 3, 3)
    mask = mask.permute(0, 3, 5, 4, 1, 2).reshape(batch, groups * 9, height, width)
    weight = value.new_ones(channels, 1, 3, 3)
    result = deform_conv2d(value.permute(0, 3, 1, 2).contiguous(), offset.contiguous(), weight,
                           padding=(1, 1), mask=mask.contiguous())
    return result.permute(0, 2, 3, 1).contiguous()


def check_reference():
    torch.manual_seed(5)
    for groups in (1, 2):
        channels = groups * 4
        value = torch.randn(2, 5, 7, channels, dtype=torch.double, requires_grad=True)
        width = ((groups * 27 + 7) // 8) * 8
        offsets = (torch.randn(2, 5, 7, width, dtype=torch.double) * .3).requires_grad_()
        args = (*ARGS[:8], groups, channels // groups, *ARGS[10:])
        actual = dcnv4_reference(value, offsets, *args)
        expected = torchvision_reference(value, offsets, groups)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        probe = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual * probe).sum(), (value, offsets))
        expected_grads = torch.autograd.grad((expected * probe).sum(), (value, offsets))
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
        assert actual_grads[1][..., groups * 27:].count_nonzero() == 0
    value = torch.randn(1, 4, 6, 32, dtype=torch.double)
    offsets = torch.zeros(1, 4, 6, 32, dtype=torch.double)
    offsets[..., 18 + 4] = -2.  # Center point, signed unnormalized weight.
    torch.testing.assert_close(dcnv4_reference(value, offsets, *ARGS), -2 * value, rtol=1e-12, atol=1e-12)
    with patch("ultralytics.nn.modules.dcnv4_backend.require_cuda_extension", side_effect=RuntimeError("missing build")):
        try:
            DCNv4Function.apply(SimpleNamespace(is_cuda=True), None, *ARGS)
        except RuntimeError as error:
            assert str(error) == "missing build"
        else:
            raise AssertionError("GPU execution must not fall back when the extension is missing")
    print("PASS: CPU core matches independent torchvision outputs and all input/offset/mask gradients, signed weights, groups and padding")


def check_pack(pack):
    path = pack / "ultralytics/nn/extra_modules/block.py"
    names = {"DCNV4_YOLO", "Bottleneck_DCNV4", "C3k_DCNv4", "C3k2_DCNv4"}
    nodes = [n for n in ast.parse(path.read_text(encoding="utf-8-sig")).body if isinstance(n, ast.ClassDef) and n.name in names]
    namespace = {"nn": torch.nn, "DCNv4": DCNv4, "Conv": Conv, "autopad": autopad,
                 "Bottleneck": Bottleneck, "C3k": C3k, "C3k2": C3k2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    a, b = namespace["C3k2_DCNv4"](128, 128, c3k=True).double(), C3k2_DCNv4(128, 128, c3k=True).double()
    b.load_state_dict(a.state_dict(), strict=True)
    for training in (False, True):
        a.train(training)
        b.train(training)
        image = torch.randn(2, 128, 5, 7, dtype=torch.double, requires_grad=True)
        left, right = a(image), b(image)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        probe = torch.randn_like(left)
        grads_a = torch.autograd.grad((left * probe).sum(), (image, *a.parameters()))
        grads_b = torch.autograd.grad((right * probe).sum(), (image, *b.parameters()))
        for x, y in zip(grads_a, grads_b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    # DCNv4 class body itself is unchanged from the supplied NCHW implementation.
    source = pack / "ultralytics/nn/extra_modules/DCNv4_op/DCNv4/modules/dcnv4.py"
    original = next(n for n in ast.parse(source.read_text(encoding="utf-8-sig")).body if isinstance(n, ast.ClassDef) and n.name == "DCNv4")
    port = next(n for n in ast.parse((ROOT / "ultralytics/nn/modules/dcnv4.py").read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "DCNv4")
    assert ast.dump(original, include_attributes=False) == ast.dump(port, include_attributes=False)
    expected = yaml_load(pack / "ultralytics/cfg/models/11/yolo11-C3k2-DCNV4.yaml")
    expected["nc"] = 1
    assert yaml_load(MODEL) == expected
    vendor = ROOT / "third_party/DCNv4_op"
    count = 0
    for file in (pack / "ultralytics/nn/extra_modules/DCNv4_op").rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
            def normalized(path):
                return "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8-sig").splitlines()).rstrip()
            assert normalized(file) == normalized(vendor / file.relative_to(pack / "ultralytics/nn/extra_modules/DCNv4_op"))
            count += 1
    print(f"PASS: complete pack YAML, unchanged DCNv4 class, C3k2 train/eval outputs/gradients and {count} vendor files matching apart from whitespace")


def check_cuda_core(device):
    compiled = require_cuda_extension()
    for batch, height, width, channels in ((1, 20, 20, 32), (4, 10, 14, 64), (1, 40, 40, 32)):
        value = torch.randn(batch, height, width, channels, device=device, requires_grad=True)
        offsets = (torch.randn(batch, height, width, 32, device=device) * .2).requires_grad_()
        args = (*ARGS[:9], channels, *ARGS[10:])
        actual = compiled.apply(value, offsets, *args)
        expected = dcnv4_reference(value, offsets, *args)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=5e-5)
        probe = torch.randn_like(actual)
        left = torch.autograd.grad((actual * probe).sum(), (value, offsets))
        right = torch.autograd.grad((expected * probe).sum(), (value, offsets))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=2e-3, atol=5e-4)
    print("PASS: actual compiled CUDA forward/backward matches bilinear reference, B=1/4, square/rectangular, C=32/64")


def check_model(output, device):
    before_cfg = yaml_load(ROOT / "ultralytics/cfg/models/11/yolo11.yaml")
    after_cfg = yaml_load(MODEL)
    before, after = before_cfg["backbone"] + before_cfg["head"], after_cfg["backbone"] + after_cfg["head"]
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [6, 8]
    net = YOLO(str(MODEL)).model.to(device)
    assert [i for i, m in enumerate(net.model) if isinstance(m, C3k2_DCNv4)] == [6, 8]
    operators = [m for m in net.modules() if isinstance(m, DCNv4)]
    assert len(operators) == 4 and [m.channels for m in operators] == [32, 32, 64, 64]
    assert all((m.group, m.kernel_size, m.stride, m.pad, m.offset_scale, m.dw_kernel_size,
                m.remove_center, m.center_feature_scale, m.without_pointwise) == (1, 3, 1, 1, 1., None, 0, False, False)
               for m in operators)
    assert type(net.model[-1]) is Detect and net.model[-1].nc == 1
    assert net.stride.tolist() == [8., 16., 32.]
    print(f"PASS: layers 6/8 only, 4 DCNv4 operators with exact pack defaults, unchanged Detect/strides; {sum(p.numel() for p in net.parameters())} parameters")
    net.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    net.train()
    batch = {"img": torch.rand(2, 3, 128, 128, device=device),
             "batch_idx": torch.tensor([0, 1], device=device), "cls": torch.zeros(2, 1, device=device),
             "bboxes": torch.tensor([[.4, .4, .15, .15], [.6, .5, .2, .2]], device=device)}
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    for step in range(2):
        optimizer.zero_grad()
        loss, items = net(batch)
        assert loss.isfinite().all()
        loss.sum().backward()
        assert all(p.grad.isfinite().all() for p in net.parameters() if p.grad is not None)
        assert all(m.offset_mask.weight.grad is not None and m.offset_mask.weight.grad.abs().max() > 0 for m in operators)
        # Raw masks start at zero, so upstream value gradients can be zero on the first step.
        if step:
            assert all(m.value_proj.weight.grad is not None and m.value_proj.weight.grad.abs().max() > 0 for m in operators)
        optimizer.step()
    print(f"PASS: {device} detection loss, two backward/AdamW steps and offset/mask/value projection gradients")
    net.eval()
    with torch.no_grad():
        for count, h, w in ((1, 640, 640), (1, 160, 224), (4, 160, 224)):
            image = torch.rand(count, 3, h, w, device=device)
            prediction = net(image)[0]
            assert prediction.shape == (count, 5, sum((h // s) * (w // s) for s in (8, 16, 32)))
            assert prediction.isfinite().all()
        expected = net(image)[0]
        fused = deepcopy(net).fuse(verbose=False)
        torch.testing.assert_close(expected, fused(image)[0], rtol=2e-3, atol=2e-4)
    checkpoint = output / "integration.pt"
    torch.save({"model": deepcopy(net).half().cpu(), "train_args": {"task": "detect"}}, checkpoint)
    loaded = YOLO(str(checkpoint)).model.to(device).float().eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, loaded(image)[0], rtol=.02, atol=.05)
    print("PASS: 640 and rectangular B=1/4 inference, Conv/BN fusion, FP16 checkpoint saving and FP32 reload")


def check_validation_table(output):
    """Check the four real validation calls and table format using synthetic metrics."""
    spec = importlib.util.spec_from_file_location("dcnv4_validation", ROOT / "val.py")
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
    assert len(rows) == 9 and rows[0] == ["验证集", "指标", "基线", "C3k2-DCNv4", "变化（百分点）"]
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
    device_id = device.split(":")[-1] if device.startswith("cuda") else "cpu"
    YOLO(str(MODEL)).train(
        data=str(yaml), epochs=1, batch=2, imgsz=128, device=device_id,
        workers=0, amp=False, optimizer="AdamW", lr0=.001, seed=0,
        project=str(output / "smoke-runs"), name="integration", exist_ok=True,
        plots=False, verbose=False, save=True, val=True,
    )
    weights = output / "smoke-runs/integration/weights/best.pt"
    assert weights.is_file()
    YOLO(str(weights)).val(data=str(yaml), split="val", imgsz=128, batch=1, workers=0,
                          device=device_id, half=False, plots=False,
                          project=str(output / "smoke-runs"), name="reload-val", exist_ok=True)
    print("PASS: synthetic full trainer, EMA, best.pt saving, reload and validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-pack", type=Path)
    parser.add_argument("--reference-only", action="store_true", help="CPU reference only; does not validate the CUDA extension")
    parser.add_argument("--device", default="1", help="CUDA device for the compiled check (server GPU1 by default)")
    parser.add_argument("--smoke-train", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = "cpu" if args.reference_only else f"cuda:{args.device}"
    if not args.reference_only:
        torch.cuda.set_device(device)
        require_cuda_extension()
    with tempfile.TemporaryDirectory(prefix="dcnv4-check-", dir=ROOT.parent) as directory:
        output = Path(directory)
        check_reference()
        if args.reference_pack:
            check_pack(args.reference_pack)
        if not args.reference_only:
            check_cuda_core(device)
        check_model(output, device)
        check_validation_table(output)
        if args.smoke_train:
            smoke_train(output, device)
    print("REFERENCE CHECKS PASSED (CUDA extension NOT tested)" if args.reference_only else "ALL DCNv4 CUDA CHECKS PASSED")


if __name__ == "__main__":
    main()
