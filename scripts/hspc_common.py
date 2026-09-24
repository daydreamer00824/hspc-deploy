"""阶段1-3 共用：确定性设置、建模、权重加载、样本切分、精度指标。"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ENC_DIR = ROOT / "hspc_encoder"
CALIB_DIR = ENC_DIR / "calib_samples"
sys.path.insert(0, str(ENC_DIR))

CALIB_N = 400          # 前 400 条：仅 INT8 校准
VAL_N = 200            # 后 200 条：仅精度验证

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def setup_determinism() -> None:
    """对齐 repro_preflight.json 的训练期数值契约。必须在任何前向之前调用。"""
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextlib.contextmanager
def math_sdpa():
    """SDPA 只用 math 后端，与训练期一致。"""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        yield


def load_config() -> dict:
    return json.loads((ENC_DIR / "model_config.json").read_text())


def build_pc(cfg: dict):
    from TransformerEncoders import PCTransformerEncoder
    return PCTransformerEncoder(
        k=cfg["point_neighbors"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        depth=cfg["depth"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=0.0,
        emb_dims=cfg["emb_dims"],
    )


def build_hsi(cfg: dict):
    from TransformerEncoders import HSITransformerEncoder
    # in_channels 类默认为 343，实际权重为 342，必须显式传参
    return HSITransformerEncoder(
        in_channels=cfg["target_bands"],
        patch_hw=cfg["patch_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        depth=cfg["depth"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=0.0,
        emb_dims=cfg["emb_dims"],
    )


def load_strict(model, ckpt_name: str):
    """strict=True 加载，任何 missing/unexpected key 都报错退出。"""
    sd = torch.load(ENC_DIR / ckpt_name, map_location="cpu", weights_only=True)
    result = model.load_state_dict(sd, strict=True)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt_name}: missing={missing} unexpected={unexpected}")
    n = sum(p.numel() for p in model.parameters())
    print(f"  [{ckpt_name}] strict load OK, {len(sd)} keys, {n:,} params")
    return model.eval()


def get_model(which: str, device="cpu"):
    cfg = load_config()
    if which == "pc":
        return load_strict(build_pc(cfg), "final_pc_encoder.pt").to(device)
    if which == "hsi":
        return load_strict(build_hsi(cfg), "final_hsi_encoder.pt").to(device)
    raise ValueError(which)


def load_samples(which: str, split: str) -> np.ndarray:
    """split: 'calib' 前400条 | 'val' 后200条 | 'all'。两者绝不重叠。"""
    fname = {"pc": "point_offsets.npy", "hsi": "hsi_patches.npy"}[which]
    arr = np.load(CALIB_DIR / fname).astype(np.float32)
    if split == "calib":
        return arr[:CALIB_N]
    if split == "val":
        return arr[CALIB_N:CALIB_N + VAL_N]
    if split == "all":
        return arr
    raise ValueError(split)


def dummy_input(which: str, batch: int) -> torch.Tensor:
    shape = {"pc": (batch, 15, 3), "hsi": (batch, 342, 3, 3)}[which]
    return torch.zeros(*shape, dtype=torch.float32)


@torch.no_grad()
def torch_forward(model, x: np.ndarray, device="cpu", batch=64) -> np.ndarray:
    out = []
    with math_sdpa():
        for i in range(0, len(x), batch):
            t = torch.from_numpy(x[i:i + batch]).to(device)
            out.append(model(t).float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------- 精度指标 ----------

def _rank_agreement(ref: np.ndarray, test: np.ndarray, k: int) -> float:
    """同模态检索自一致性：以 ref 的近邻排序为基准，test 的 top-k 命中率。

    对应 inference.py:score_window 对余弦做 argmax 的下游行为。
    """
    r = F.normalize(torch.from_numpy(ref).float(), dim=1)
    t = F.normalize(torch.from_numpy(test).float(), dim=1)
    Sr, St = r @ r.T, t @ t.T
    Sr.fill_diagonal_(-2.0)
    St.fill_diagonal_(-2.0)
    ref_top1 = Sr.argmax(1)
    test_topk = St.topk(k, dim=1).indices
    return float((test_topk == ref_top1[:, None]).any(1).float().mean())


def margin_stats(ref: np.ndarray) -> dict:
    """基准嵌入自身的 top1-top2 余弦裕度分布，用于解释排序一致率为何达不到 100%。"""
    r = F.normalize(torch.from_numpy(ref).float(), dim=1)
    S = r @ r.T
    S.fill_diagonal_(-2.0)
    top2 = S.topk(2, dim=1).values
    m = top2[:, 0] - top2[:, 1]
    return {
        "margin_mean": float(m.mean()),
        "margin_median": float(m.median()),
        "margin_p10": float(m.quantile(0.1)),
        "margin_min": float(m.min()),
    }


def accuracy_report(ref: np.ndarray, test: np.ndarray) -> dict:
    """三件套：cos-sim 均值/最小、max abs err、Top-1/Top-5 排序一致率。"""
    assert ref.shape == test.shape, f"{ref.shape} vs {test.shape}"
    a = torch.from_numpy(ref).float()
    b = torch.from_numpy(test).float()
    cos = F.cosine_similarity(a, b, dim=1)
    abs_err = (a - b).abs()
    return {
        "n": int(ref.shape[0]),
        "cos_sim_mean": float(cos.mean()),
        "cos_sim_min": float(cos.min()),
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
        "rel_l2": float(((a - b).norm() / a.norm())),
        "top1_agreement": _rank_agreement(ref, test, 1),
        "top5_agreement": _rank_agreement(ref, test, 5),
    }


def print_report(tag: str, rep: dict) -> None:
    print(f"  {tag:<28} cos_mean={rep['cos_sim_mean']:.6f} cos_min={rep['cos_sim_min']:.6f} "
          f"max_abs={rep['max_abs_err']:.3e} top1={rep['top1_agreement']:.3f} top5={rep['top5_agreement']:.3f}")
