"""测试环境公共配置。"""

from __future__ import annotations

import os


os.environ.setdefault("CSDM_panel_DISABLE_LLM_AUTO", "1")
