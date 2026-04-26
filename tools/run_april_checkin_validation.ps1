param(
    [string]$PythonExe = "python"
)

$arguments = @(
    "-m", "pytest",
    "tests/test_realtime_inference.py",
    "tests/test_v4_delay_predictor.py",
    "-q",
    "--basetemp", ".pytest_tmp/submission",
    "-o", "cache_dir=.pytest_tmp/cache"
)

Write-Host "Running April check-in validation tests..."
& $PythonExe @arguments
