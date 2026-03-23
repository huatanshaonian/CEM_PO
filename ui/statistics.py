"""
RCS 统计分析模块

所有统计量在线性域 (m²) 下计算，显示时转换为 dBsm。
"""
import numpy as np


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


def compute_comparison_statistics(rcs_db_a, rcs_db_b):
    """
    计算两条 RCS 数据之间的对比统计量。

    输入: rcs_db_a, rcs_db_b — dBsm 数组（等长）
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
    mean_diff_lin = float(np.mean(diff_lin))
    max_abs_diff_lin = float(np.max(np.abs(diff_lin)))

    stats = {
        'N':               len(a),
        'rmse_lin':        rmse_lin,
        'mean_diff_lin':   mean_diff_lin,
        'max_abs_diff_lin': max_abs_diff_lin,
        # 相关系数（线性域）
        'correlation':     float(np.corrcoef(a_lin, b_lin)[0, 1]) if len(a) > 1 else float('nan'),
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
    ('mean_diff_lin',    'Mean Diff (m²)',      '{:+.4e}',   'mean_diff_lin'),
    ('max_abs_diff_lin', 'Max |Diff| (m²)',     '{:.4e}',    'max_abs_diff_lin'),
    ('correlation',      'Correlation',         '{:.6f}',    'correlation'),
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
