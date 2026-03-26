import os
import glob
import random

import numpy as np
from scipy.io import savemat

try:
    import wfdb
except ImportError as e:
    raise ImportError(
        "本脚本依赖 wfdb 包，请先安装：\n"
        "    pip install wfdb\n"
    ) from e


def collect_records(afdb_dir):
    """
    收集 AFDB 中所有记录名（不带扩展名）
    """
    dat_files = glob.glob(os.path.join(afdb_dir, "*.dat"))
    records = sorted(
        {os.path.splitext(os.path.basename(p))[0] for p in dat_files}
    )
    return records


def segment_record(record_path_no_ext, win_sec=10.0, stride_sec=10.0, channel=0):
    """
    从单条记录中切片，并根据标注打 AF / 非 AF 标签。

    返回:
        X: [N, 1, L]
        y_true: [N]，0=非AF, 1=AF
    """
    # 读信号
    record = wfdb.rdrecord(record_path_no_ext, channels=[channel])
    fs = record.fs
    sig = record.p_signal[:, 0]  # [T]

    # 读标注（atr 通道）
    try:
        ann = wfdb.rdann(record_path_no_ext, 'atr')
    except Exception:
        # 没有 atr 标注就全部当作未标注非 AF
        win_size = int(fs * win_sec)
        stride = int(fs * stride_sec)
        segments = []
        labels = []
        for start in range(0, len(sig) - win_size + 1, stride):
            end = start + win_size
            seg = sig[start:end]
            segments.append(seg.astype(np.float32)[None, :])
            labels.append(0)
        if not segments:
            return np.empty((0, 1, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        X = np.stack(segments, axis=0)
        y_true = np.asarray(labels, dtype=np.int32)
        return X, y_true

    # AFDB 中 AF 节律一般使用 aux_note，如 '(AFIB', '(AFL' 等
    af_indices = set()
    for sample, aux in zip(ann.sample, ann.aux_note):
        if aux is None:
            continue
        aux_str = aux.decode() if isinstance(aux, bytes) else str(aux)
        if "AFIB" in aux_str or "AFL" in aux_str:
            af_indices.add(sample)

    win_size = int(fs * win_sec)
    stride = int(fs * stride_sec)

    segments = []
    labels = []

    for start in range(0, len(sig) - win_size + 1, stride):
        end = start + win_size
        seg = sig[start:end]

        # 判断这个窗口内是否出现 AF 标注
        is_af = any(start <= idx < end for idx in af_indices)
        label = 1 if is_af else 0

        segments.append(seg.astype(np.float32)[None, :])  # [1, L]
        labels.append(label)

    if not segments:
        return np.empty((0, 1, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X = np.stack(segments, axis=0)  # [N,1,L]
    y_true = np.asarray(labels, dtype=np.int32)
    return X, y_true


def build_dataset(
    afdb_dir,
    out_dir,
    win_sec=10.0,
    stride_sec=10.0,
    channel=0,
    train_ratio=0.8,
    labeled_positive_ratio=0.1,
    seed=42,
):
    """
    从 AFDB 原始文件构建 SSL+nnPU 所需的 .mat 数据：
        - afdb_train.mat: X, y_pu, y, prior
        - afdb_test.mat:  X, y
    """
    random.seed(seed)
    np.random.seed(seed)

    records = collect_records(afdb_dir)
    if not records:
        raise RuntimeError(f"在目录 {afdb_dir} 中未找到任何 .dat 记录文件")

    print(f"找到 {len(records)} 条记录，开始切片...")

    all_segments = []
    all_labels = []
    for rec in records:
        rec_path_no_ext = os.path.join(afdb_dir, rec)
        print(f"  处理记录 {rec} ...")
        X_rec, y_rec = segment_record(
            rec_path_no_ext,
            win_sec=win_sec,
            stride_sec=stride_sec,
            channel=channel,
        )
        if X_rec.shape[0] == 0:
            continue
        all_segments.append(X_rec)
        all_labels.append(y_rec)

    if not all_segments:
        raise RuntimeError("未从任何记录中切出有效片段，请检查 win_sec / stride_sec 等参数。")

    X = np.concatenate(all_segments, axis=0)  # [N,1,L]
    y_true = np.concatenate(all_labels, axis=0)  # [N]

    print(f"总样本数: {X.shape[0]}, 其中 AF 数量: {(y_true == 1).sum()}, 非 AF 数量: {(y_true == 0).sum()}")

    # 计算类先验 P(Y=1)
    prior = float((y_true == 1).mean())
    print(f"估计类先验 prior = P(Y=1) ≈ {prior:.4f}")

    # 构造 PU 标签：部分 AF 标为 1，其余全部记为 -1
    pos_idx = np.where(y_true == 1)[0]
    n_pos = len(pos_idx)
    n_labeled_pos = max(1, int(n_pos * labeled_positive_ratio))
    labeled_pos_idx = np.random.choice(pos_idx, size=n_labeled_pos, replace=False)

    y_pu = -np.ones_like(y_true, dtype=np.int32)
    y_pu[labeled_pos_idx] = 1

    # 打乱并划分 train/test
    N = X.shape[0]
    indices = np.random.permutation(N)
    split = int(N * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X[train_idx]
    y_train_true = y_true[train_idx]
    y_train_pu = y_pu[train_idx]

    X_test = X[test_idx]
    y_test_true = y_true[test_idx]

    os.makedirs(out_dir, exist_ok=True)

    train_mat_path = os.path.join(out_dir, "afdb_train.mat")
    test_mat_path = os.path.join(out_dir, "afdb_test.mat")

    savemat(
        train_mat_path,
        {
            "X": X_train,
            "y": y_train_true,
            "y_pu": y_train_pu,
            "prior": np.array([prior], dtype=np.float32),
        },
    )
    savemat(
        test_mat_path,
        {
            "X": X_test,
            "y": y_test_true,
        },
    )

    print(f"训练集已保存到: {train_mat_path}")
    print(f"测试集已保存到: {test_mat_path}")


if __name__ == "__main__":
    # 默认假设本文件位于工程根目录下的 ecg_ssl_pu 子目录
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    AFDB_DIR = os.path.join(ROOT_DIR, "mit-bih-atrial-fibrillation-database-1.0.0")
    OUT_DIR = os.path.join(ROOT_DIR, "ecg_ssl_pu", "data")

    build_dataset(
        afdb_dir=AFDB_DIR,
        out_dir=OUT_DIR,
        win_sec=10.0,
        stride_sec=10.0,
        channel=0,
        train_ratio=0.8,
        labeled_positive_ratio=1.0,  # 将所有 AF 都标为正样本以缓解极端不平衡
        seed=42,
    )



