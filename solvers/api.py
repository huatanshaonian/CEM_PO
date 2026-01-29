from .po import DiscretePOIntegrator
from .ribbon import TrueRibbonIntegrator, AnalyticRibbonIntegrator

# 可用算法列表
AVAILABLE_ALGORITHMS = {
    'discrete_po_none': {
        'name': '离散PO (无校正)',
        'class': DiscretePOIntegrator,
        'kwargs': {'sinc_mode': 'none'},
        'description': '纯离散 PO，无 sinc 校正。需要最多网格点，精度依赖网格密度。',
    },
    'discrete_po_sinc_u': {
        'name': '离散PO (单向Sinc)',
        'class': DiscretePOIntegrator,
        'kwargs': {'sinc_mode': 'u_only'},
        'description': '离散 PO + u方向 sinc 校正。原始 Ribbon 近似，适中精度。',
    },
    'discrete_po_sinc_dual': {
        'name': '离散PO (双向Sinc)',
        'class': DiscretePOIntegrator,
        'kwargs': {'sinc_mode': 'dual'},
        'description': '离散 PO + 双向 sinc 校正。最佳精度，适合斜入射场景。',
    },
    'gauss_ribbon': {
        'name': 'Gauss-Ribbon (自适应)',
        'class': TrueRibbonIntegrator,
        'kwargs': {},
        'description': 'v方向离散，u方向自适应Gauss积分。高效且精确，推荐用于非多项式曲面。',
    },
    'analytic_ribbon': {
        'name': '解析Ribbon (论文算法)',
        'class': AnalyticRibbonIntegrator,
        'kwargs': {},
        'description': '严格按照1995论文实现：多项式拟合+精确阴影边界+解析积分。',
    },
}

def get_integrator(algorithm='discrete_po_sinc_dual', **kwargs):
    """
    算法工厂函数：根据名称获取积分器实例

    参数:
        algorithm: 算法名称
    返回:
        积分器实例
    """
    if algorithm not in AVAILABLE_ALGORITHMS:
        raise ValueError(f"未知算法: {algorithm}. 可用: {list(AVAILABLE_ALGORITHMS.keys())}")

    algo_info = AVAILABLE_ALGORITHMS[algorithm]
    merged_kwargs = {**algo_info.get('kwargs', {}), **kwargs}
    return algo_info['class'](**merged_kwargs)

def list_algorithms():
    """列出所有可用算法"""
    result = []
    for key, info in AVAILABLE_ALGORITHMS.items():
        result.append({
            'id': key,
            'name': info['name'],
            'description': info['description']
        })
    return result
