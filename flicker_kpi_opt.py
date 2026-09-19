"""
Flicker KPI threshold 最佳化
============================
輸入兩類已標記的序列：
  - 不閃 (label 0)：不閃 folder 的全部序列 + 混合 folder 中標為不閃者
  - 閃   (label 1)：混合 folder 中標為閃者

判定規則（序列層級）:
  某 metric m 有 >= K 個位置 value > t_m  ->  該序列判為「閃」；20 項 metric 任一成立即閃 (OR)
  等價於: s_m(K) = 該 metric map 第 K 大的值,  閃 <=> 存在 m 使 s_m(K) > t_m
  （超標幅度越大、超標位置越多，s_m(K) 越大 -> 越容易判閃）
  K 以 pixel 計；block-based 換算為 ceil(K / block_size^2) 個 block（面積一致）。

最佳化目標: 漏抓 = 0 的前提下，誤報（不閃序列被判閃）最少。
  1. 每個 metric 先把門檻放在所有不閃序列之上 (t_m > max_clean_m) -> 誤報 0
  2. 仍未被抓到的閃序列，需把某些 metric 的門檻降到其 s 值以下；每個「降門檻」選項
     會連帶誤報一些不閃序列。求「覆蓋全部未抓閃序列、且誤報序列聯集最小」:
       - scipy.optimize.milp 可用時解 0/1 整數規劃（精確解）
       - 否則用 greedy set cover + 反向刪除（近似解）
  3. 誤報集合確定後，以 max-min margin 決定門檻位置（log 域）:
       max δ  s.t.  每個未誤報的不閃序列  s_im <= t_m e^-δ  (所有 m)
                    每個閃序列            存在 m 使 s_jm >= t_m e^+δ
     令 t_m = max_clean_m · e^δ，可行性對 δ 單調 -> 二分搜尋。
     分離度 = e^δ*：所有閃序列 S/t >= e^δ*，所有未誤報不閃序列 S/t <= e^-δ*。
  3'. 嚴格模式（不漏抓優先）: t_m = max_clean_m · (1 + tol)
     每個 metric 貼著未誤報不閃序列的上緣，任何一項稍微超出不閃樣本就觸發。
     tol 若會讓某個已知閃序列漏抓，自動降到仍可全部抓到的最大值（回報 tol_applied）。
對每個候選 K 各解一次，取誤報最少者；
  同誤報數時：平衡模式取分離度最大者，嚴格模式取「閃序列最小 S/t」最大者。
"""

import math

import numpy as np

from flicker_kpi_core import (
    METRIC_NAMES, load_frames, compute_metric_maps, k_for_level,
)

LEVELS = ("pixel", "block")
KEYS = [(lv, m) for lv in LEVELS for m in METRIC_NAMES]      # 20 項，固定順序
DEFAULT_K_LIST = (1, 4, 16, 64, 256)


# --------------------------------------------------------------------------
# 1. 每組序列只需保留每個 metric 的前 Kmax 大值
# --------------------------------------------------------------------------
def sequence_topk(paths, kmax, diff_mode="peak_to_peak", block_size=2):
    """
    讀一組序列，回傳 {(level, metric): 由大到小的前 k 個值 (float32)}。
    只存 top-k，不保留整張 map（批次掃大量 1080p 序列時省記憶體）。
    """
    stack = load_frames(paths)
    out = {}
    for lv, blk in (("pixel", 1), ("block", block_size)):
        k = k_for_level(kmax, lv, block_size)
        for m, arr in compute_metric_maps(stack, block=blk, diff_mode=diff_mode).items():
            flat = arr.ravel()
            kk = min(k, flat.size)
            top = np.partition(flat, flat.size - kk)[flat.size - kk:]
            out[(lv, m)] = np.sort(top)[::-1].astype(np.float32)
    return out


def stat_matrix(topks, K, block_size=2):
    """topks: [dict] -> S (n_seq, 20)，S[i, m] = 第 K 大值（不足 K 個位置時為 0）"""
    S = np.zeros((len(topks), len(KEYS)), np.float64)
    for i, tk in enumerate(topks):
        for j, key in enumerate(KEYS):
            k = k_for_level(K, key[0], block_size)
            v = tk[key]
            S[i, j] = v[k - 1] if len(v) >= k else 0.0
    return S


# --------------------------------------------------------------------------
# 2. 最佳化
# --------------------------------------------------------------------------
def _gap_threshold(lo, hi):
    """lo < t < hi，取幾何中點（metric 皆非負；lo = 0 時取 hi/2）"""
    if hi <= 0:
        return -1e-6                    # 所有值皆 0 時仍需 > 判定成立
    if lo <= 0:
        return hi / 2.0
    return math.sqrt(lo * hi)


def _solve_cover(options, U, n_clean):
    """
    options: [(metric_idx, cut, cover_set, fa_set)]
    求選一組 options 覆蓋 U 且 |∪fa| 最小。回傳 (選中 index list, exact: bool)
    """
    if not U:
        return [], True
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
        nO = len(options)
        nV = nO + n_clean
        c = np.concatenate([np.full(nO, 1e-4), np.ones(n_clean)])   # 次要: 少降門檻
        U_list = sorted(U)
        A = lil_matrix((len(U_list) + sum(len(o[3]) for o in options), nV))
        lb, ub = [], []
        r = 0
        for j in U_list:                                  # 覆蓋: Σ x_o >= 1
            for oi, o in enumerate(options):
                if j in o[2]:
                    A[r, oi] = 1
            lb.append(1); ub.append(np.inf); r += 1
        for oi, o in enumerate(options):                  # 誤報: y_i - x_o >= 0
            for i in o[3]:
                A[r, oi] = -1
                A[r, nO + i] = 1
                lb.append(0); ub.append(np.inf); r += 1
        res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub),
                   integrality=np.ones(nV), bounds=Bounds(0, 1))
        if res.success:
            return [oi for oi in range(nO) if res.x[oi] > 0.5], True
    except ImportError:
        pass

    # greedy set cover（以新增誤報數為成本）+ 反向刪除
    chosen, covered, fa = [], set(), set()
    while covered != U:
        best, best_key = None, None
        for oi, (_, _, cov, f) in enumerate(options):
            new_cov = len((cov & U) - covered)
            if new_cov == 0:
                continue
            new_fa = len(f - fa)
            key = (new_fa / new_cov, -new_cov)
            if best_key is None or key < best_key:
                best, best_key = oi, key
        if best is None:
            break
        chosen.append(best)
        covered |= options[best][2] & U
        fa |= options[best][3]
    for oi in list(reversed(chosen)):
        rest = [o for o in chosen if o != oi]
        cov = set().union(*(options[o][2] for o in rest)) if rest else set()
        if U <= cov:
            chosen = rest
    return chosen, False


def optimize_thresholds(S, labels, mode="strict", tol=0.0):
    """
    S: (n, 20) 序列統計量；labels: (n,) bool，True = 閃
    回傳 dict:
      thresholds (20,), fa_idx, miss_idx, exact, separation,
      seq_ratio (n,)  = max_m S/t（>1 判閃）, seq_metric (n,) = 造成該最大值的 metric index
    """
    labels = np.asarray(labels, bool)
    ci, fi = np.flatnonzero(~labels), np.flatnonzero(labels)
    if len(ci) == 0:
        raise ValueError("需要至少一組不閃序列")
    C, F = S[ci], S[fi]
    maxc = C.max(axis=0)

    covered0 = (F > maxc).any(axis=1) if len(fi) else np.zeros(0, bool)
    U = set(np.flatnonzero(~covered0).tolist())               # index into fi

    options = []
    for m in range(S.shape[1]):
        for cut in sorted({float(F[j, m]) for j in U}, reverse=True):
            cov = {j for j in U if F[j, m] >= cut}
            fa = set(np.flatnonzero(C[:, m] >= cut).tolist())  # index into ci
            options.append((m, cut, cov, fa))
    chosen, exact = _solve_cover(options, U, len(ci))

    fa_set = set()
    for oi in chosen:
        fa_set |= options[oi][3]
    keep = [k for k in range(len(ci)) if k not in fa_set]      # 未誤報的不閃序列
    maxc2 = C[keep].max(axis=0) if keep else np.zeros(S.shape[1])
    base = np.maximum(maxc2, 1e-6)                             # 避免 log(0)

    tol_applied = None
    if mode == "strict":
        # 閃序列需 F > t；tol 過大會漏抓時，二分搜尋仍可全抓的最大 tol
        def covers(d):
            return bool((F > base * math.exp(d)).any(axis=1).all()) if len(fi) else True
        d_want = math.log1p(max(tol, 0.0))
        if covers(d_want):
            d_lo = d_want
        else:
            d_lo, d_hi = 0.0, d_want
            for _ in range(60):
                mid = (d_lo + d_hi) / 2
                if covers(mid):
                    d_lo = mid
                else:
                    d_hi = mid
        tol_applied = math.expm1(d_lo)
        t = base * math.exp(d_lo)
    else:
        def feasible(d):
            return bool((F >= base * math.exp(2 * d)).any(axis=1).all())

        with np.errstate(divide="ignore"):
            d_hi = float(np.max(np.log(np.maximum(F, 1e-12) / base)) / 2) if len(fi) else 0.0
        d_lo = 0.0
        if d_hi > 0 and feasible(d_hi):
            d_lo = d_hi
        else:
            for _ in range(60):
                mid = (d_lo + d_hi) / 2
                if feasible(mid):
                    d_lo = mid
                else:
                    d_hi = mid
        t = base * math.exp(d_lo)

    # 門檻無條件進位到小數 4 位（GUI 顯示/輸入用），並以主視窗相同的 float32 比較驗證
    # 判定不變；metric map 為 float32，判定式為 value > threshold
    S32 = S.astype(np.float32)

    def predict(th):
        return (S32 > np.asarray(th, np.float64).astype(np.float32)).any(axis=1)

    tq = np.ceil(t * 1e4) / 1e4
    if np.array_equal(predict(tq), predict(t)):
        t = tq

    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(t > 0, S / t, np.where(S > t, np.inf, 0.0))
    seq_ratio = R.max(axis=1)
    seq_metric = R.argmax(axis=1)
    pred = predict(t)
    fa_idx = ci[pred[ci]].tolist()
    miss_idx = fi[~pred[fi]].tolist()

    ok_clean = [i for i in ci if not pred[i]]
    sep_c = 1 / seq_ratio[ok_clean].max() if ok_clean and seq_ratio[ok_clean].max() > 0 else np.inf
    sep_f = seq_ratio[fi].min() if len(fi) else np.inf
    # 造成誤報的 metric（誤報序列在該 metric 超過門檻）
    fa_metrics = sorted({m for i in fa_idx for m in np.flatnonzero(S[i] > t)})
    return {"thresholds": t, "fa_idx": fa_idx, "miss_idx": miss_idx, "exact": exact,
            "separation": float(min(sep_c, sep_f)), "seq_ratio": seq_ratio,
            "seq_metric": seq_metric, "lowered": fa_metrics,
            "flick_min": float(sep_f),                                  # 閃序列最小 S/t
            "clean_max": float(1 / sep_c) if np.isfinite(sep_c) else 0.0,  # 未誤報不閃最大 S/t
            "mode": mode, "tol": float(tol), "tol_applied": tol_applied}


def sweep_k(topks, labels, k_list=DEFAULT_K_LIST, block_size=2,
            modes=("strict", "balanced"), tol=0.0):
    """
    每個 (模式, K) 各最佳化一次。回傳依模式分組、組內依
    (誤報數, 漏抓數, -閃序列最小 S/t [嚴格] / -分離度 [平衡]) 排序的結果 list
    """
    results = []
    for mode in modes:
        group = []
        for K in k_list:
            S = stat_matrix(topks, K, block_size)
            r = optimize_thresholds(S, labels, mode, tol)
            r["K"] = int(K)
            r["S"] = S
            group.append(r)
        score = (lambda r: r["flick_min"]) if mode == "strict" else (lambda r: r["separation"])
        group.sort(key=lambda r: (len(r["fa_idx"]), len(r["miss_idx"]), -score(r)))
        results += group
    return results


def thresholds_to_dict(t):
    out = {"pixel": {}, "block": {}}
    for (lv, m), v in zip(KEYS, t):
        out[lv][m] = float(v)
    return out
