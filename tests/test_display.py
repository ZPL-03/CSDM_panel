"""展示精度与结构化输入分离。"""

import pytest

from core.display import format_number, format_parameter


@pytest.mark.parametrize(("value", "expected"), [
    (None, "-"), (45.0, "45"), (1.46, "1.5"), (-3.558, "-3.6"),
    (-0.0, "0"), (0.0049, "4.9e-03"), (-0.0049, "-4.9e-03"),
])
def test_summary_number(value, expected):
    assert format_number(value) == expected


def test_geometry_and_layup_precision():
    assert format_parameter("panel_length_mm", 600.123) == "600.1"
    assert format_parameter("skin_thickness_mm", 0.125) == "0.125"
    assert format_parameter("skin_f0", 0.286) == "0.286"
    assert format_parameter("alpha_deg", -45.0) == "-45"
