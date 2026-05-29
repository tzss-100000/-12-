$ErrorActionPreference = "Stop"
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 --version
    if ($LASTEXITCODE -eq 0) {
        & py -3.12 -m venv .venv
    } else {
        & python -m venv .venv
    }
} else {
    & python -m venv .venv
}
$python = ".\.venv\Scripts\python.exe"
$version = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($version -notmatch "^(3\.9|3\.10|3\.11|3\.12)$") {
    throw "Python $version is not suitable for TensorFlow. Please install Python 3.12 and run again."
}
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
Write-Host "Environment is ready. Activate with: .\.venv\Scripts\Activate.ps1"
