param([switch]$CheckEnvironment)

# conda run 负责设置 AGENT 的 DLL 搜索路径，不直接启动环境目录中的 python.exe。
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$conda = Get-Command conda -CommandType Application -ErrorAction Stop | Select-Object -First 1
$entry = if ($CheckEnvironment) { 'scripts/check_env.py' } else { 'main.py' }
Push-Location -LiteralPath $root
try {
    & $conda.Source run -n AGENT --no-capture-output python -X utf8 -B $entry
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
