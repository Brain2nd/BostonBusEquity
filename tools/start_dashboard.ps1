param(
    [string]$PythonExe = "python",
    [string]$Bundle = "models/delay_predictor_v4_score_best_online_safe_bundle.joblib",
    [string]$Host = "127.0.0.1",
    [int]$Port = 8000
)

$arguments = @(
    "-m", "src.inference.serve",
    "--bundle", $Bundle,
    "--host", $Host,
    "--port", $Port
)

Write-Host "Starting dashboard with bundle: $Bundle"
Write-Host "Open http://$Host`:$Port/ after the server starts."

& $PythonExe @arguments
