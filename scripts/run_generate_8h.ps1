<#
PowerShell script to run the Duffing data generator for a specified number of minutes
and save output with today's date in the filename.

Usage examples (PowerShell):
  # Run for 8 hours, compute lyapunov (slow):
  .\run_generate_8h.ps1 -Minutes 480 -ComputeLyapunov

  # Run for 8 hours, do NOT compute lyapunov (faster):
  .\run_generate_8h.ps1 -Minutes 480
#>

param(
    [int]$Minutes = 480,
    [switch]$ComputeLyapunov,
    [string]$OutPrefix = "duffing_dataset"
)

# compute date string YYYYMMDD
$date = (Get-Date).ToString('yyyyMMdd')
$outFile = "${OutPrefix}_${date}.csv"

Write-Host "Starting generation at" (Get-Date) "-> output:" $outFile
Write-Host "Run time (minutes):" $Minutes
if ($ComputeLyapunov) { Write-Host "Lyapunov computation: ENABLED (this may be slow)" } else { Write-Host "Lyapunov computation: disabled" }

# If you use a virtualenv, ensure it's activated first. If not, this will use whatever 'python' is on PATH.
# Example to activate .venv in repo root (uncomment if needed):
# & "${PWD}\.venv\Scripts\Activate.ps1"

$computeFlag = if ($ComputeLyapunov) { '--compute-lyapunov' } else { '' }

# Build the command. We pass --minutes to run for approx that many minutes.
$cmd = @(
    'python', '-m', 'duffing.generate_data',
    '--n', '1000000', # large n (ignored when --minutes is set)
    '--out', $outFile,
    '--seed', '0',
    '--minutes', $Minutes.ToString()
)

if ($ComputeLyapunov) { $cmd += '--compute-lyapunov' }

# Run the command and stream output
Write-Host "Running: $($cmd -join ' ')"
$process = Start-Process -FilePath $cmd[0] -ArgumentList $cmd[1..($cmd.Length - 1)] -NoNewWindow -Wait -PassThru

if ($process.ExitCode -eq 0) {
    Write-Host "Generation finished successfully at" (Get-Date)
    Write-Host "Saved to" $outFile
} else {
    Write-Host "Generation exited with code" $process.ExitCode -ForegroundColor Red
}
