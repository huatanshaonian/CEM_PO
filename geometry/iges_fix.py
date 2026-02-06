"""
IGES 格式预处理：修复 Tecplot 等工具生成的非标准 IGES 文件。

问题：某些生成器的 P 段每行只放一个数值且不带逗号分隔符，
也缺少实体记录终止分号，导致 OCC 等 CAD 库无法读取。

修复：
  1. P 段单值行补尾逗号
  2. 每个实体最后一行的尾逗号改为分号
"""

import os
import tempfile


def fix_iges(input_path: str, output_path: str | None = None) -> str:
    """
    修复非标准 IGES 文件的 P 段格式。

    参数:
        input_path:  输入 IGES 文件路径
        output_path: 输出路径，None 则写入临时文件

    返回:
        输出文件的路径
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.igs', prefix='iges_fixed_')
        os.close(fd)

    # 检测原文件换行符风格
    with open(input_path, 'rb') as f:
        chunk = f.read(4096)
    line_ending = '\r\n' if b'\r\n' in chunk else '\n'

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # ---- 第一遍：从 D 段收集每个实体的 P 段末行号 ----
    last_p_seq: set[int] = set()
    d_line_buf = None

    for line in lines:
        if len(line) >= 73 and line[72] == 'D':
            if d_line_buf is None:
                d_line_buf = line
            else:
                try:
                    p_start = int(d_line_buf[8:16].strip())
                    p_count = int(line[24:32].strip())
                    last_p_seq.add(p_start + p_count - 1)
                except ValueError:
                    pass
                d_line_buf = None

    # ---- 第二遍：修复 P 段行 ----
    out_lines: list[str] = []

    for line in lines:
        if len(line) >= 73 and line[72] == 'P':
            data = line[:64]
            tail = line[64:]     # cols 65 起保持不变

            try:
                p_seq = int(line[73:80].strip())
            except ValueError:
                out_lines.append(line)
                continue

            stripped = data.rstrip()
            is_last = p_seq in last_p_seq

            if is_last:
                if stripped.endswith(';'):
                    pass
                elif stripped.endswith(','):
                    stripped = stripped[:-1] + ';'
                else:
                    stripped += ';'
            else:
                if not stripped.endswith(',') and not stripped.endswith(';'):
                    stripped += ','

            out_lines.append(stripped.ljust(64) + tail)
        else:
            out_lines.append(line)

    with open(output_path, 'w', encoding='utf-8', newline=line_ending) as f:
        f.writelines(out_lines)

    return output_path


def needs_fix(filepath: str) -> bool:
    """
    快速检测 IGES 文件的 P 段是否缺少标准分隔符。
    只检查前几行 P 段数据，避免读取整个大文件。
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        p_count = 0
        for line in f:
            if len(line) >= 73 and line[72] == 'P':
                stripped = line[:64].rstrip()
                if stripped and not stripped.endswith(',') and not stripped.endswith(';'):
                    return True
                p_count += 1
                if p_count >= 20:
                    break
    return False
