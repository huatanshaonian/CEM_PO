"""
PTD 算法注册表 - 仿照 solvers/api.py 的设计风格

通过算法 ID 切换 PTD 实现, 现有调度代码 (solvers/ptd.py, rcs_analyzer.py,
freq_sweep.py) 通过 get_ptd_function 获取具体计算函数。

新增 PTD 算法只需在此添加一项注册即可。
"""
from physics.ptd_core import compute_ptd_contribution
from physics.mec_core import compute_mec_contribution
from physics.mec_truncated_core import compute_mec_truncated_contribution

PTD_ALGORITHMS = {
    'ufimtsev_eew': {
        'name': 'Ufimtsev EEW',
        'func': compute_ptd_contribution,
        'description': 'Ufimtsev fringe wave (Eq. 7.137) + FG_monostatic 衍射系数',
        'supports_cross_pol': False,
    },
    'michaeli_mec': {
        'name': 'Michaeli MEC (1986 Part I)',
        'func': compute_mec_contribution,
        'description': 'Ray-coordinate 等效边电流闭式, 全方位无奇异 (除 Ufimtsev 奇点)',
        'supports_cross_pol': True,
    },
    'michaeli_mec_truncated': {
        'name': 'Michaeli MEC + Johansen 1996 截断修正',
        'func': compute_mec_truncated_contribution,
        'description': '非截断 Michaeli MEC 减去 Johansen 1996 Eq.26/27 修正, '
                       '消除 Ufimtsev 奇点 + 改进掠射区行为',
        'supports_cross_pol': True,
    },
}

DEFAULT_PTD_ALGORITHM = 'ufimtsev_eew'


def get_ptd_function(algo_id):
    """根据算法 ID 获取 PTD 计算函数。"""
    if algo_id not in PTD_ALGORITHMS:
        raise ValueError(
            f"未知 PTD 算法: '{algo_id}'. 可用: {list(PTD_ALGORITHMS.keys())}")
    return PTD_ALGORITHMS[algo_id]['func']


def supports_cross_polarization(algo_id):
    """判断指定算法是否支持交叉极化 (VH/HV)。"""
    if algo_id not in PTD_ALGORITHMS:
        return False
    return PTD_ALGORITHMS[algo_id].get('supports_cross_pol', False)


def list_ptd_algorithms():
    """列出所有可用 PTD 算法 (供 UI / CLI 显示)。"""
    return [
        {
            'id': key,
            'name': info['name'],
            'description': info['description'],
            'supports_cross_pol': info.get('supports_cross_pol', False),
        }
        for key, info in PTD_ALGORITHMS.items()
    ]
