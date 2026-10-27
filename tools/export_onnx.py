"""Export the trained Module 1 and Module 2 networks to ONNX for the website,
without needing PyTorch at runtime.

The "Your panels" page runs these models in the visitor's browser with
onnxruntime-web, so photographs never leave their device.

How: a PyTorch checkpoint is a zip of raw tensor buffers plus a pickle that
says how to assemble them. This script reads the tensors with plain Python
and NumPy, then writes the MobileNetV2 graph (torchvision layout) directly
in ONNX, folding each batch-norm into the convolution before it.

Checked against the predictions PyTorch itself saved during evaluation
(module1_detection/outputs/test_predictions.csv and
module2_powerloss/outputs/test_predictions.csv) on the same test photographs.

Writes website/models/{soiling_classifier,powerloss_cnn}.onnx and meta.json.

Usage:  python -m tools.export_onnx
"""
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website" / "models"
M1_CKPT = ROOT / "module1_detection" / "outputs" / "mobilenetv2_soiling.pt"
M2_CKPT = ROOT / "module2_powerloss" / "outputs" / "powerloss_cnn.pt"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG = 224
ENV_FEATURES = ["irradiance", "tod_sin", "tod_cos"]


# ---- reading a .pt checkpoint without torch -------------------------------------------
_DTYPES = {"FloatStorage": np.float32, "LongStorage": np.int64, "IntStorage": np.int32,
           "HalfStorage": np.float16, "DoubleStorage": np.float64, "BoolStorage": np.bool_}


class _Storage:
    def __init__(self, name):
        self.dtype = _DTYPES[name]


def _rebuild_tensor(storage, offset, size, stride, *_):
    item = storage.itemsize
    arr = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size), strides=[s * item for s in stride])
    return np.array(arr)


def load_pt(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if module == "torch" and name in _DTYPES:
                return _Storage(name)
            if module == "collections" and name == "OrderedDict":
                from collections import OrderedDict
                return OrderedDict
            raise pickle.UnpicklingError(f"unexpected class {module}.{name}")

        def persistent_load(self, pid):
            _, storage_type, key, _, numel = pid
            raw = z.read(f"{prefix}/data/{key}")
            return np.frombuffer(raw, dtype=storage_type.dtype, count=numel)

    return Unpickler(z.open(f"{prefix}/data.pkl")).load()


# ---- MobileNetV2 as an ONNX graph -------------------------------------------------------
SETTINGS = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


class Graph:
    def __init__(self, sd):
        self.sd, self.nodes, self.inits, self.n = sd, [], [], 0

    def name(self, base):
        self.n += 1
        return f"{base}_{self.n}"

    def const(self, arr, base):
        """Weights are stored at half precision (halves the download) and cast
        back to float32 inside the graph, so all arithmetic stays float32."""
        nm = self.name(base)
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float32 and arr.size > 64:
            self.inits.append(numpy_helper.from_array(arr.astype(np.float16), nm + "_h"))
            self.nodes.append(helper.make_node("Cast", [nm + "_h"], [nm], to=TensorProto.FLOAT))
        else:
            self.inits.append(numpy_helper.from_array(arr, nm))
        return nm

    def op(self, op_type, inputs, **attrs):
        out = self.name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def conv_bn(self, x, conv, bn, stride=1, groups=1, relu6=True):
        w = self.sd[f"{conv}.weight"].astype(np.float32)
        g, b = self.sd[f"{bn}.weight"], self.sd[f"{bn}.bias"]
        m, v = self.sd[f"{bn}.running_mean"], self.sd[f"{bn}.running_var"]
        scale = g / np.sqrt(v + 1e-5)
        wf = w * scale[:, None, None, None]
        bf = b - m * scale
        k = w.shape[-1]
        y = self.op("Conv", [x, self.const(wf, "w"), self.const(bf, "b")],
                    kernel_shape=[k, k], strides=[stride, stride], pads=[k // 2] * 4, group=groups)
        if relu6:
            y = self.op("Clip", [y, self.const(np.array(0, np.float32), "lo"), self.const(np.array(6, np.float32), "hi")])
        return y

    def linear(self, x, prefix):
        return self.op("Gemm", [x, self.const(self.sd[f"{prefix}.weight"], "W"), self.const(self.sd[f"{prefix}.bias"], "B")], transB=1)

    def backbone(self, x):
        f = "features"
        x = self.conv_bn(x, f"{f}.0.0", f"{f}.0.1", stride=2)
        idx, inp = 1, 32
        for t, c, n, s in SETTINGS:
            for i in range(n):
                stride = s if i == 0 else 1
                p = f"{f}.{idx}.conv"
                h = x
                if t == 1:
                    h = self.conv_bn(h, f"{p}.0.0", f"{p}.0.1", stride=stride, groups=inp)
                    h = self.conv_bn(h, f"{p}.1", f"{p}.2", relu6=False)
                else:
                    hidden = inp * t
                    h = self.conv_bn(h, f"{p}.0.0", f"{p}.0.1")
                    h = self.conv_bn(h, f"{p}.1.0", f"{p}.1.1", stride=stride, groups=hidden)
                    h = self.conv_bn(h, f"{p}.2", f"{p}.3", relu6=False)
                x = self.op("Add", [x, h]) if stride == 1 and inp == c else h
                inp, idx = c, idx + 1
        x = self.conv_bn(x, f"{f}.18.0", f"{f}.18.1")
        x = self.op("GlobalAveragePool", [x])
        return self.op("Flatten", [x], axis=1)

    def model(self, inputs, outputs, doc):
        g = helper.make_graph(self.nodes, doc, inputs, outputs, self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="solara")
        m.ir_version = 8
        onnx.checker.check_model(m)
        return m


def build_classifier(sd):
    g = Graph(sd)
    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, IMG, IMG])
    x = g.backbone("image")
    x = g.op("Relu", [g.linear(x, "classifier.1")])
    logits = g.linear(x, "classifier.4")
    g.nodes[-1].output[0] = "logits"
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 4])
    return g.model([img], [out], "soiling_classifier")


def build_regressor(sd):
    g = Graph(sd)
    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, IMG, IMG])
    env = helper.make_tensor_value_info("env", TensorProto.FLOAT, ["batch", 3])
    f = g.backbone("image")
    e = g.op("Relu", [g.linear("env", "env.0")])
    z = g.op("Concat", [f, e], axis=1)
    z = g.op("Relu", [g.linear(z, "head.1")])
    z = g.op("Sigmoid", [g.linear(z, "head.4")])
    g.nodes.append(helper.make_node("Squeeze", [z, g.const(np.array([1], np.int64), "ax")], ["loss"]))
    out = helper.make_tensor_value_info("loss", TensorProto.FLOAT, ["batch"])
    return g.model([img, env], [out], "powerloss_cnn")


# ---- check against PyTorch's own saved predictions ---------------------------------------
def preprocess(path):
    im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    a = (a - MEAN) / STD
    return a.transpose(2, 0, 1).astype(np.float32)


def verify(n=48):
    s1 = ort.InferenceSession(str(OUT / "soiling_classifier.onnx"))
    s2 = ort.InferenceSession(str(OUT / "powerloss_cnn.onnx"))
    names = json.loads((OUT / "meta.json").read_text())["class_names"]

    p1 = pd.read_csv(ROOT / "module1_detection" / "outputs" / "test_predictions.csv").sample(n, random_state=1)
    x = np.stack([preprocess(ROOT / p) for p in p1.path])
    logits = s1.run(None, {"image": x})[0]
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    same = (prob.argmax(1) == p1.predicted.to_numpy()).mean()
    conf = np.abs(prob.max(1) - p1.confidence.to_numpy()).max()
    print(f"classifier vs PyTorch: same class {same:.1%}, max confidence difference {conf:.4f}")

    idx = pd.read_csv(ROOT / "data" / "deepsolareye_index.csv", usecols=["path", "timestamp", "irradiance"])
    p2 = pd.read_csv(ROOT / "module2_powerloss" / "outputs" / "test_predictions.csv").sample(n, random_state=1)
    p2 = p2.merge(idx[["path", "timestamp"]], on="path")
    ts = pd.to_datetime(p2.timestamp)
    frac = (ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600) / 24
    env = np.stack([p2.irradiance, np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)], 1).astype(np.float32)
    x = np.stack([preprocess(ROOT / p) for p in p2.path])
    loss = s2.run(None, {"image": x, "env": env})[0] * 100
    diff = np.abs(loss - p2.cnn.to_numpy())
    print(f"regressor vs PyTorch: max difference {diff.max():.3f} points, mean {diff.mean():.4f}")
    print(f"class names: {names}")
    for f in sorted(OUT.glob("*.onnx")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c1, c2 = load_pt(M1_CKPT), load_pt(M2_CKPT)
    onnx.save(build_classifier(c1["state_dict"]), OUT / "soiling_classifier.onnx")
    onnx.save(build_regressor(c2["state_dict"]), OUT / "powerloss_cnn.onnx")
    meta = {
        "class_names": c1["class_names"], "loss_bins_pct": c1["loss_bins"], "img_size": IMG,
        "mean": MEAN, "std": STD, "env_features": ENV_FEATURES,
        "note": "Trained on DeepSolarEye (one panel, fixed camera). Irradiance is DeepSolarEye's normalised value (about 0-1).",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    verify()


if __name__ == "__main__":
    main()
