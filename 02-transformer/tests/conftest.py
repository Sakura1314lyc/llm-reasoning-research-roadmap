"""让测试可以导入上一级目录中的教学模块。"""

from pathlib import Path
import sys


TRANSFORMER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRANSFORMER_ROOT))
