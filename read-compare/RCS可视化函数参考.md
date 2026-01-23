# RCS可视化系统参考手册

## 📖 概述

本文档是简化后RCS可视化系统的完整参考，基于两个核心模块提供RCS数据读取、处理和可视化功能。

## 🗂️ 项目结构（简化版）

```
F:\data\wavelet\
├── rcs_data_reader.py          # 数据读取和处理模块 ⭐
├── rcs_visual.py               # 可视化功能模块 ⭐
└── RCS可视化函数参考.md        # 本文档
```

**核心优势**：
- **模块化设计**：数据处理与可视化分离
- **功能完整**：支持所有类型的RCS分析和可视化
- **易于扩展**：清晰的接口设计便于二次开发
- **高效性能**：优化的数据处理算法

---

## 🔧 核心模块详解

### 📊 rcs_data_reader.py - 数据读取模块

**主要功能**：
- 多编码格式CSV数据读取
- 自适应矩阵尺寸处理
- NaN值插值和数据清洗
- 参数数据加载
- 线性值与分贝值转换

#### 🎯 核心函数

##### 1. `get_adaptive_rcs_matrix()` - 自适应矩阵读取 ⭐

**函数签名**:
```python
def get_adaptive_rcs_matrix(model_id="001", freq_suffix="1.5G",
                           data_dir=r"F:\data\parameter\csv_output"):
```

**功能特点**:
- 支持任意尺寸的RCS数据（不限于91×91）
- 自动检测数据维度
- 智能数据插值和清理
- 完整的统计信息输出

**返回数据结构**:
```python
{
    'rcs_linear': ndarray,      # RCS线性值矩阵
    'rcs_db': ndarray,          # RCS分贝值矩阵
    'theta_values': ndarray,    # 俯仰角数组
    'phi_values': ndarray,      # 偏航角数组
    'theta_grid': ndarray,      # 俯仰角网格
    'phi_grid': ndarray,        # 偏航角网格
    'data_info': dict          # 详细统计信息
}
```

**使用示例**:
```python
import rcs_data_reader as rdr

# 基本使用
data = rdr.get_adaptive_rcs_matrix("001", "1.5G")
rcs_linear = data['rcs_linear']
rcs_db = data['rcs_db']

# 获取不同模型和频率
data_002 = rdr.get_adaptive_rcs_matrix("002", "3G")
data_custom = rdr.get_adaptive_rcs_matrix("005", "2.4G", "custom/path")

# 访问统计信息
info = data['data_info']
print(f"矩阵尺寸: {info['matrix_shape']}")
print(f"有效数据点: {info['valid_points']}")
print(f"RCS范围: {info['rcs_linear_range']}")
```

##### 2. `load_parameters()` - 参数数据读取

**函数签名**:
```python
def load_parameters(params_file, verbose=True):
```

**功能**：
- 读取设计参数CSV文件
- 多编码格式自动检测
- NaN值智能填充
- 数据验证和清理

**使用示例**:
```python
# 加载参数数据
param_data, param_names = rdr.load_parameters("F:/data/parameter/parameters_sorted.csv")
print(f"参数数量: {len(param_names)}")
print(f"数据形状: {param_data.shape}")
```

##### 3. `load_single_rcs_data()` - 单文件RCS读取

**函数签名**:
```python
def load_single_rcs_data(data_dir, model_id, freq_suffix, verbose=True):
```

**功能**：
- 读取单个RCS数据文件
- 数据验证和预处理
- 兼容性处理

**使用示例**:
```python
# 读取单个RCS文件
rcs_data = rdr.load_single_rcs_data("F:/data/parameter/csv_output", "001", "1.5G")
```

---

### 🎨 rcs_visual.py - 可视化模块

**主要功能**：
- 2D热图可视化（线性值+分贝值）
- 3D表面图可视化
- 球坐标3D可视化
- 多模型对比功能
- 数据保存和加载

#### 🎯 核心函数

##### 1. `plot_2d_heatmap()` - 2D热图可视化 ⭐

**函数签名**:
```python
def plot_2d_heatmap(model_id="001", freq_suffix="1.5G", data_dir=None,
                   db_vmin=None, db_vmax=None, linear_vmin=None, linear_vmax=None,
                   figsize=(16, 6), save_path=None, show_plot=True):
```

**功能特点**：
- 同时显示线性值和分贝值热图
- 自定义colorbar范围
- 正确的坐标系显示（phi为X轴，theta为Y轴，小角度在上）
- jet颜色映射
- 高质量图像输出

**坐标系说明**：
- **X轴**: 偏航角(Phi) [-45°, +45°]
- **Y轴**: 俯仰角(Theta) [45°在上, 135°在下]
- **颜色**: jet映射（蓝→绿→黄→红）

**使用示例**:
```python
import rcs_visual as rv

# 基本使用
fig, axes = rv.plot_2d_heatmap("001", "1.5G")

# 自定义分贝范围突出强散射区域
fig, axes = rv.plot_2d_heatmap("001", "1.5G", db_vmin=-25, db_vmax=-5)

# 自定义线性值范围
fig, axes = rv.plot_2d_heatmap("001", "1.5G", linear_vmin=1e-4, linear_vmax=1e-1)

# 保存到指定文件
fig, axes = rv.plot_2d_heatmap("001", "1.5G", save_path="custom_heatmap.png")
```

##### 2. `plot_3d_surface()` - 3D表面图 ⭐

**函数签名**:
```python
def plot_3d_surface(model_id="001", freq_suffix="1.5G", data_dir=None,
                   db_vmin=None, db_vmax=None, figsize=(12, 8),
                   save_path=None, show_plot=True):
```

**功能特点**：
- 3D表面显示RCS分布
- 可调节观察角度
- 颜色编码表示RCS强度
- 高质量3D渲染

**使用示例**:
```python
# 基本3D图
fig, ax = rv.plot_3d_surface("001", "1.5G")

# 自定义分贝范围
fig, ax = rv.plot_3d_surface("001", "1.5G", db_vmin=-30, db_vmax=-10)

# 修改观察角度
fig, ax = rv.plot_3d_surface("001", "1.5G")
ax.view_init(elev=45, azim=60)
```

##### 3. `plot_spherical_3d()` - 球坐标3D图

**函数签名**:
```python
def plot_spherical_3d(model_id="001", freq_suffix="1.5G", data_dir=None,
                     db_vmin=None, db_vmax=None, figsize=(10, 10),
                     save_path=None, show_plot=True):
```

**功能特点**：
- 球坐标系3D可视化
- 径向距离表示RCS强度
- 适合分析全向散射特性

**使用示例**:
```python
# 球坐标可视化
fig, ax = rv.plot_spherical_3d("001", "1.5G")

# 自定义范围
fig, ax = rv.plot_spherical_3d("001", "1.5G", db_vmin=-35, db_vmax=-15)
```

##### 4. `plot_all_views()` - 综合可视化

**函数签名**:
```python
def plot_all_views(model_id="001", freq_suffix="1.5G", data_dir=None,
                  db_vmin=None, db_vmax=None, save_prefix=None):
```

**功能**：
- 一次性生成所有视图类型
- 统一的colorbar范围
- 批量文件保存

**使用示例**:
```python
# 生成所有视图
rv.plot_all_views("001", "1.5G", save_prefix="model_001_analysis")
```

##### 5. `compare_models()` - 多模型对比

**函数签名**:
```python
def compare_models(model_ids, freq_suffix="1.5G", data_dir=None,
                  db_vmin=None, db_vmax=None, figsize=(20, 12),
                  save_path=None, show_plot=True):
```

**功能**：
- 多个模型并排对比显示
- 统一的颜色标度
- 差异分析

**使用示例**:
```python
# 对比多个模型
models = ["001", "002", "003", "004"]
fig, axes = rv.compare_models(models, "1.5G")

# 自定义对比范围
fig, axes = rv.compare_models(models, "1.5G", db_vmin=-30, db_vmax=-10)
```

##### 6. `get_rcs_matrix()` - 数据接口

**函数签名**:
```python
def get_rcs_matrix(model_id="001", freq_suffix="1.5G", data_dir=None):
```

**功能**：
- 直接获取处理好的矩阵数据
- 无可视化的纯数据接口
- 便于后续分析

**使用示例**:
```python
# 获取矩阵数据
data = rv.get_rcs_matrix("001", "1.5G")
rcs_linear = data['rcs_linear']
rcs_db = data['rcs_db']

# 进行自定义分析
max_rcs = np.nanmax(rcs_linear)
max_pos = np.unravel_index(np.nanargmax(rcs_linear), rcs_linear.shape)
```

---

## 🔄 模块协作机制

### 数据流向
```
CSV文件 → rcs_data_reader.py → 处理后的矩阵数据 → rcs_visual.py → 可视化结果
```

### 典型工作流程

#### 1. 基础数据分析
```python
import rcs_data_reader as rdr
import rcs_visual as rv
import numpy as np

# 步骤1: 读取数据
data = rdr.get_adaptive_rcs_matrix("001", "1.5G")

# 步骤2: 基础分析
print(f"数据形状: {data['rcs_linear'].shape}")
print(f"RCS范围: {data['data_info']['rcs_db_range']} dB")

# 步骤3: 可视化
fig_2d, axes = rv.plot_2d_heatmap("001", "1.5G")
fig_3d, ax = rv.plot_3d_surface("001", "1.5G")
```

#### 2. 多模型对比分析
```python
# 模型列表
models = ["001", "002", "003"]

# 数据读取和预处理
model_data = {}
for model in models:
    model_data[model] = rdr.get_adaptive_rcs_matrix(model, "1.5G")

# 对比可视化
rv.compare_models(models, "1.5G", db_vmin=-30, db_vmax=-10)

# 定量对比
for i, model1 in enumerate(models):
    for model2 in models[i+1:]:
        corr = np.corrcoef(
            model_data[model1]['rcs_linear'].flatten(),
            model_data[model2]['rcs_linear'].flatten()
        )[0,1]
        print(f"{model1} vs {model2} 相关系数: {corr:.3f}")
```

#### 3. 频率特性分析
```python
# 同一模型不同频率
frequencies = ["1.5G", "2.4G", "3G"]
model_id = "001"

freq_data = {}
for freq in frequencies:
    freq_data[freq] = rdr.get_adaptive_rcs_matrix(model_id, freq)
    # 生成对应的可视化
    rv.plot_2d_heatmap(model_id, freq, save_path=f"rcs_{model_id}_{freq}.png")

# 频率相关性分析
for i, freq1 in enumerate(frequencies):
    for freq2 in frequencies[i+1:]:
        corr = np.corrcoef(
            freq_data[freq1]['rcs_db'].flatten(),
            freq_data[freq2]['rcs_db'].flatten()
        )[0,1]
        print(f"{freq1} vs {freq2} 频率相关性: {corr:.3f}")
```

---

## 📊 高级应用案例

### 1. 自定义分析函数
```python
def analyze_rcs_characteristics(model_id, freq_suffix):
    """综合RCS特性分析"""
    # 获取数据
    data = rdr.get_adaptive_rcs_matrix(model_id, freq_suffix)
    rcs_db = data['rcs_db']

    # 计算关键指标
    max_rcs = np.nanmax(rcs_db)
    min_rcs = np.nanmin(rcs_db)
    mean_rcs = np.nanmean(rcs_db)
    std_rcs = np.nanstd(rcs_db)

    # 方向性分析
    directivity = max_rcs - min_rcs

    # 主瓣方向
    max_pos = np.unravel_index(np.nanargmax(rcs_db), rcs_db.shape)
    main_theta = data['theta_values'][max_pos[0]]
    main_phi = data['phi_values'][max_pos[1]]

    # 强散射区域比例
    threshold = mean_rcs + std_rcs
    strong_scatter_ratio = np.sum(rcs_db > threshold) / np.sum(~np.isnan(rcs_db))

    return {
        'max_rcs_db': max_rcs,
        'min_rcs_db': min_rcs,
        'mean_rcs_db': mean_rcs,
        'std_rcs_db': std_rcs,
        'directivity_db': directivity,
        'main_lobe_direction': (main_theta, main_phi),
        'strong_scatter_ratio': strong_scatter_ratio
    }

# 使用示例
analysis = analyze_rcs_characteristics("001", "1.5G")
print(f"方向性: {analysis['directivity_db']:.1f} dB")
print(f"主瓣方向: θ={analysis['main_lobe_direction'][0]:.1f}°, φ={analysis['main_lobe_direction'][1]:.1f}°")
```

### 2. 批量处理pipeline
```python
def batch_process_models(model_list, freq_list, output_dir="results"):
    """批量处理多个模型和频率"""
    import os

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for model in model_list:
        results[model] = {}
        for freq in freq_list:
            print(f"处理模型 {model}, 频率 {freq}")

            # 数据读取
            data = rdr.get_adaptive_rcs_matrix(model, freq)

            # 生成所有可视化
            save_prefix = os.path.join(output_dir, f"model_{model}_{freq}")
            rv.plot_all_views(model, freq, save_prefix=save_prefix)

            # 保存矩阵数据
            rv.save_rcs_matrix(data, f"{save_prefix}_matrix.npz")

            # 特性分析
            analysis = analyze_rcs_characteristics(model, freq)
            results[model][freq] = analysis

    return results

# 使用示例
models = ["001", "002", "003"]
frequencies = ["1.5G", "2.4G", "3G"]
batch_results = batch_process_models(models, frequencies)
```

### 3. 数据质量检查
```python
def data_quality_check(model_id, freq_suffix):
    """数据质量检查和报告"""
    data = rdr.get_adaptive_rcs_matrix(model_id, freq_suffix)

    info = data['data_info']
    rcs_linear = data['rcs_linear']
    rcs_db = data['rcs_db']

    # 基础检查
    total_points = rcs_linear.size
    valid_points = info['valid_points']
    completeness = valid_points / total_points * 100

    # 数据范围检查
    linear_range = info['rcs_linear_range']
    db_range = info['rcs_db_range']

    # 异常值检查
    q1, q3 = np.nanpercentile(rcs_db, [25, 75])
    iqr = q3 - q1
    outlier_threshold_low = q1 - 1.5 * iqr
    outlier_threshold_high = q3 + 1.5 * iqr
    outliers = np.sum((rcs_db < outlier_threshold_low) | (rcs_db > outlier_threshold_high))

    # 生成报告
    report = {
        'model_id': model_id,
        'frequency': freq_suffix,
        'data_completeness': completeness,
        'total_points': total_points,
        'valid_points': valid_points,
        'linear_range': linear_range,
        'db_range': db_range,
        'outlier_count': outliers,
        'quality_score': min(100, completeness * (1 - outliers/valid_points))
    }

    return report

# 使用示例
quality_report = data_quality_check("001", "1.5G")
print(f"数据完整性: {quality_report['data_completeness']:.1f}%")
print(f"质量评分: {quality_report['quality_score']:.1f}")
```

---

## 📁 文件输出说明

### 自动生成文件
- **2D热图**: `rcs_heatmap_{model}_{freq}.png`
- **3D表面图**: `rcs_3d_surface_{model}_{freq}.png`
- **球坐标图**: `rcs_spherical_{model}_{freq}.png`
- **矩阵数据**: `rcs_matrix_{model}_{freq}.npz`

### 文件特性
- **分辨率**: 300 DPI
- **格式**: PNG (图像), NPZ (数据)
- **命名规则**: 一致的命名约定便于批量处理

---

## ⚠️ 使用注意事项

### 1. 数据路径配置
```python
# 默认数据路径
DEFAULT_DATA_DIR = r"F:\data\parameter\csv_output"

# 如需修改，可以在函数调用时指定
data = rdr.get_adaptive_rcs_matrix("001", "1.5G", data_dir="your/custom/path")
```

### 2. 内存使用优化
```python
# 对于大批量处理，及时释放内存
import gc

for model in large_model_list:
    data = rdr.get_adaptive_rcs_matrix(model, "1.5G")
    # 处理数据...
    del data
    gc.collect()  # 强制垃圾回收
```

### 3. 错误处理
```python
try:
    data = rdr.get_adaptive_rcs_matrix("999", "10G")  # 可能不存在的数据
except FileNotFoundError:
    print("数据文件不存在，请检查模型ID和频率")
except Exception as e:
    print(f"数据读取错误: {e}")
```

---

## 🚀 快速开始模板

```python
# 完整的RCS分析脚本模板
import rcs_data_reader as rdr
import rcs_visual as rv
import numpy as np

def main():
    # 配置
    model_id = "001"
    freq_suffix = "1.5G"

    # 1. 数据读取
    print("读取RCS数据...")
    data = rdr.get_adaptive_rcs_matrix(model_id, freq_suffix)

    # 2. 基础信息
    info = data['data_info']
    print(f"数据形状: {info['matrix_shape']}")
    print(f"有效数据点: {info['valid_points']}")
    print(f"RCS范围: {info['rcs_db_range'][0]:.1f} - {info['rcs_db_range'][1]:.1f} dB")

    # 3. 可视化
    print("生成可视化...")

    # 2D热图
    fig_2d, axes = rv.plot_2d_heatmap(model_id, freq_suffix,
                                     save_path=f"heatmap_{model_id}_{freq_suffix}.png")

    # 3D表面图
    fig_3d, ax = rv.plot_3d_surface(model_id, freq_suffix,
                                    save_path=f"surface3d_{model_id}_{freq_suffix}.png")

    # 球坐标图
    fig_sph, ax_sph = rv.plot_spherical_3d(model_id, freq_suffix,
                                           save_path=f"spherical_{model_id}_{freq_suffix}.png")

    # 4. 数据分析
    rcs_db = data['rcs_db']
    max_rcs = np.nanmax(rcs_db)
    max_pos = np.unravel_index(np.nanargmax(rcs_db), rcs_db.shape)
    max_theta = data['theta_values'][max_pos[0]]
    max_phi = data['phi_values'][max_pos[1]]

    print(f"最大RCS: {max_rcs:.1f} dB")
    print(f"最大RCS位置: θ={max_theta:.1f}°, φ={max_phi:.1f}°")

    # 5. 保存数据
    rv.save_rcs_matrix(data, f"data_{model_id}_{freq_suffix}.npz")

    print("分析完成！")

if __name__ == "__main__":
    main()
```

---

## 📝 版本信息

- **系统版本**: v2.0 简化版
- **更新日期**: 2024年
- **核心模块**: `rcs_data_reader.py` + `rcs_visual.py`
- **主要改进**:
  - 删除冗余代码，简化项目结构
  - 优化模块职责分工
  - 提高代码复用性和维护性
  - 完善的错误处理和数据验证

---

**💡 提示**: 本系统设计为模块化架构，便于扩展和定制。如需添加新功能，建议保持数据处理与可视化的分离原则。