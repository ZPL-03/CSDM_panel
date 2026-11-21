"""Windows PowerShell 启动入口的环境、错误传播和命令路由。"""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="需要 Windows PowerShell")
@pytest.mark.parametrize("check_environment", [False, True])
def test_startup_routes_through_conda_and_preserves_exit_status(tmp_path, check_environment):
    root = Path(__file__).resolve().parents[1]
    script = str(root / "scripts/start.ps1").replace("'", "''")
    option = "-CheckEnvironment" if check_environment else ""
    # 模拟 conda 进程，核对执行时的错误策略、工作目录和参数，不启动 GUI 或依赖检查。
    command = """
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$ErrorActionPreference = 'Continue'
function Get-Command {
    param($Name, $CommandType, $ErrorAction)
    [pscustomobject]@{Source = {
        [ordered]@{error_action=$ErrorActionPreference; cwd=(Get-Location).Path; arguments=$args} | ConvertTo-Json -Compress
        $global:LASTEXITCODE = 23
    }}
}
& 'SCRIPT' OPTION
exit $LASTEXITCODE
""".replace("SCRIPT", script).replace("OPTION", option)
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                            cwd=tmp_path, capture_output=True, encoding="utf-8", timeout=30)
    assert result.returncode == 23, result.stderr
    invocation = json.loads(result.stdout)
    assert invocation["error_action"] == "Stop"
    assert Path(invocation["cwd"]) == root
    assert invocation["arguments"] == ["run", "-n", "AGENT", "--no-capture-output", "python", "-X", "utf8", "-B",
                                       "scripts/check_env.py" if check_environment else "main.py"]
