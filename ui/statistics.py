"""
RCS 统计分析模块

所有统计量在线性域 (m²) 下计算，显示时转换为 dBsm。
峰值匹配统计基于 dB 域峰检测 + 最近邻配对。
"""
import numpy as np
from scipy.signal import find_peaks


def db_to_linear(rcs_db):
    """dBsm → m²"""
    rcs_db = np.asarray(rcs_db, dtype=float)
    return 10.0 ** (rcs_db / 10.0)


def linear_to_db(rcs_lin):
    """m² → dBsm"""
    return 10.0 * np.log10(np.maximum(rcs_lin, 1e-30))


def compute_statistics(rcs_db):
    """
    计算单条 RCS 数据的统计量。

    输入: rcs_db — dBsm 数组
    返回: dict，包含各统计量（线性域计算，dB 域显示）
    """
    rcs_db = np.asarray(rcs_db, dtype=float)
    valid = np.isfinite(rcs_db)
    if not valid.any():
        return None

    rcs_db = rcs_db[valid]
    rcs_lin = db_to_linear(rcs_db)

    stats = {
        'N':          len(rcs_db),
        'mean_lin':   float(np.mean(rcs_lin)),
        'mean_db':    float(linear_to_db(np.mean(rcs_lin))),
        'median_lin': float(np.median(rcs_lin)),
        'median_db':  float(linear_to_db(np.median(rcs_lin))),
        'max_lin':    float(np.max(rcs_lin)),
        'max_db':     float(np.max(rcs_db)),
        'min_lin':    float(np.min(rcs_lin)),
        'min_db':     float(np.min(rcs_db)),
        'std_lin':    float(np.std(rcs_lin)),
        'rms_lin':    float(np.sqrt(np.mean(rcs_lin ** 2))),
        'rms_db':     float(linear_to_db(np.sqrt(np.mean(rcs_lin ** 2)))),
        'dynamic_range_db': float(np.max(rcs_db) - np.min(rcs_db)),
    }
    return stats


def _detect_and_match_peaks(rcs_db_ref, rcs_db_test, prominence=3.0, max_shift=None):
    """
    检测两条 RCS 曲线的峰值并做最近邻配对。

    参数:
        rcs_db_ref:  参考曲线 (dBsm)
        rcs_db_test: 被测曲线 (dBsm)
        prominence:  峰检测最小突出量 (dB)
        max_shift:   最大配对距离（采样点数），默认 N//10

    返回:
        dict: peak_match_rate, mean_pos_shift, mean_amp_err_db,
              n_peaks_ref, n_peaks_test, n_matched
    """
    N = len(rcs_db_ref)
    if max_shift is None:
        max_shift = max(N // 10, 3)

    peaks_ref, props_ref = find_peaks(rcs_db_ref, prominence=prominence)
    peaks_test, _ = find_peaks(rcs_db_test, prominence=prominence)

    n_ref = len(peaks_ref)
    n_test = len(peaks_test)

    if n_ref == 0:
        return {
            'peak_match_rate': float('nan'),
            'mean_pos_shift': float('nan'),
            'mean_amp_err_db': float('nan'),
            'n_peaks_ref': 0, 'n_peaks_test': n_test, 'n_matched': 0,
        }

    # 最近邻配对：对每个参考峰找最近的测试峰
    matched_shifts = []
    matched_amp_errs = []
    used_test = set()

    for rp in peaks_ref:
        if len(peaks_test) == 0:
            break
        dists = np.abs(peaks_test - rp)
        order = np.argsort(dists)
        for idx in order:
            if dists[idx] > max_shift:
                break
            if idx not in used_test:
                tp = peaks_test[idx]
                used_test.add(idx)
                matched_shifts.append(abs(int(tp) - int(rp)))
                matched_amp_errs.append(abs(float(rcs_db_test[tp] - rcs_db_ref[rp])))
                break

    n_matched = len(matched_shifts)
    match_rate = n_matched / n_ref if n_ref > 0 else 0.0
    mean_shift = float(np.mean(matched_shifts)) if n_matched > 0 else float('nan')
    mean_amp_err = float(np.mean(matched_amp_errs)) if n_matched > 0 else float('nan')

    return {
        'peak_match_rate': match_rate,
        'mean_pos_shift': mean_shift,
        'mean_amp_err_db': mean_amp_err,
        'n_peaks_ref': n_ref,
        'n_peaks_test': n_test,
        'n_matched': n_matched,
    }


def compute_comparison_statistics(rcs_db_a, rcs_db_b):
    """
    计算两条 RCS 数据之间的对比统计量。

    输入: rcs_db_a (参考), rcs_db_b (被测) — dBsm 数组（等长）
    返回: dict
    """
    a = np.asarray(rcs_db_a, dtype=float)
    b = np.asarray(rcs_db_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return None

    a, b = a[valid], b[valid]
    a_lin, b_lin = db_to_linear(a), db_to_linear(b)

    diff_lin = a_lin - b_lin
    rmse_lin = float(np.sqrt(np.mean(diff_lin ** 2)))
    max_abs_diff_lin = float(np.max(np.abs(diff_lin)))
    # 均值差: 线性域分别求均值，转dB后相减 = 10*log10(mean(A)/mean(B))
    mean_a_db = 10.0 * np.log10(max(np.mean(a_lin), 1e-30))
    mean_b_db = 10.0 * np.log10(max(np.mean(b_lin), 1e-30))
    mean_diff_db = float(mean_a_db - mean_b_db)

    # 峰值匹配统计（a 为参考）
    peak_stats = _detect_and_match_peaks(a, b)

    stats = {
        'N':               len(a),
        'rmse_lin':        rmse_lin,
        'mean_diff_db':    mean_diff_db,
        'max_abs_diff_lin': max_abs_diff_lin,
        'correlation':     float(np.corrcoef(a_lin, b_lin)[0, 1]) if len(a) > 1 else float('nan'),
        # 峰值匹配
        'peak_match_rate': peak_stats['peak_match_rate'],
        'mean_pos_shift':  peak_stats['mean_pos_shift'],
        'mean_amp_err_db': peak_stats['mean_amp_err_db'],
        'n_peaks_ref':     peak_stats['n_peaks_ref'],
        'n_peaks_test':    peak_stats['n_peaks_test'],
        'n_matched':       peak_stats['n_matched'],
    }
    return stats


# ── 表格格式化 ──

SINGLE_STAT_ROWS = [
    ('N',               'Sample Count',        '{:d}',      'N'),
    ('mean_db',         'Mean (dBsm)',         '{:.2f}',    'mean_db'),
    ('mean_lin',        'Mean (m²)',           '{:.4e}',    'mean_lin'),
    ('median_db',       'Median (dBsm)',       '{:.2f}',    'median_db'),
    ('max_db',          'Max (dBsm)',          '{:.2f}',    'max_db'),
    ('min_db',          'Min (dBsm)',          '{:.2f}',    'min_db'),
    ('rms_db',          'RMS (dBsm)',          '{:.2f}',    'rms_db'),
    ('std_lin',         'Std Dev (m²)',        '{:.4e}',    'std_lin'),
    ('dynamic_range_db','Dynamic Range (dB)',  '{:.2f}',    'dynamic_range_db'),
]

COMPARE_STAT_ROWS = [
    ('N',                'Sample Count',        '{:d}',      'N'),
    ('rmse_lin',         'RMSE (m²)',           '{:.4e}',    'rmse_lin'),
    ('mean_diff_db',     'Mean Diff (dB)',      '{:+.2f}',   'mean_diff_db'),
    ('max_abs_diff_lin', 'Max |Diff| (m²)',     '{:.4e}',    'max_abs_diff_lin'),
    ('correlation',      'Correlation',         '{:.6f}',    'correlation'),
    ('peak_match_rate',  'Peak Match Rate',     '{:.1%}',    'peak_match_rate'),
    ('n_matched',        'Matched/Ref Peaks',   '{:d}',      'n_matched'),
    ('n_peaks_ref',      'Ref Peaks',           '{:d}',      'n_peaks_ref'),
    ('mean_pos_shift',   'Mean Peak Shift (pts)','{:.1f}',   'mean_pos_shift'),
    ('mean_amp_err_db',  'Mean Peak Err (dB)',  '{:.2f}',    'mean_amp_err_db'),
]


def format_stats_table(stats_dict, row_defs):
    """将统计 dict 按 row_defs 格式化为 [(label, value_str), ...] 列表。"""
    rows = []
    for _, label, fmt, key in row_defs:
        val = stats_dict.get(key)
        if val is None:
            rows.append((label, 'N/A'))
        elif isinstance(val, float) and not np.isfinite(val):
            rows.append((label, 'N/A'))
        else:
            rows.append((label, fmt.format(val)))
    return rows
