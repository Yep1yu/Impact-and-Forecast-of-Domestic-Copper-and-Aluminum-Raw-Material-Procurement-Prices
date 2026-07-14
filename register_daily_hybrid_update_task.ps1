param(
    [string]$TaskName = "BSH Changjiang Hybrid Price Update",
    [string]$Python = "python",
    [string]$ProjectDir = "C:\Users\Administrator\Documents\BSHintern"
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $ProjectDir "update_hybrid_price_database.py"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Cannot find update script: $scriptPath"
}

$action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "update_hybrid_price_database.py" `
    -WorkingDirectory $ProjectDir

$triggers = @(
    (New-ScheduledTaskTrigger -Daily -At "10:30"),
    (New-ScheduledTaskTrigger -Daily -At "11:30"),
    (New-ScheduledTaskTrigger -Daily -At "14:30")
)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $triggers `
    -Settings $settings `
    -Description "Fetch CCMN Changjiang prices, rebuild daily and monthly forecasts, and update the Streamlit dashboard database." `
    -Force

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Make sure CCMN_COOKIE is available as a Windows user or system environment variable."
