# Register the V4 live recorder + watchdog as scheduled tasks.
# RUN THIS SCRIPT FROM AN ELEVATED POWERSHELL (Run as Administrator).
# No password is required: tasks run as you, at log on.
#
# Properties of the registered tasks:
#   - Run as: your own account (not SYSTEM)
#   - Run level: Limited (NOT highest privileges)
#   - Trigger: At log on (no password required; starts when you sign in)
#   - Execution time limit: none (PT0S)
#   - Battery: allowed to start and keep running on battery
#   - Restart on failure: 3 attempts, 1 minute apart (plus the wrapper's own
#     infinite restart loop inside each script)

$ErrorActionPreference = "Stop"

$repo = "D:\Work\Attempts to Code\BTC Strategy\BTC Strategy"
$user = "$env:USERDOMAIN\$env:USERNAME"

$common = @{
    Trigger     = New-ScheduledTaskTrigger -AtLogOn -User $user
    Settings    = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)
    RunLevel    = "Limited"
    Force       = $true
}

$recorderAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$repo\scripts\run_live_recorder.ps1`""
$watchdogAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$repo\scripts\run_live_watchdog.ps1`""

Register-ScheduledTask -TaskName "V4LiveRecorder" -Action $recorderAction @common
Register-ScheduledTask -TaskName "V4LiveWatchdog" -Action $watchdogAction @common

Write-Host "Registered V4LiveRecorder and V4LiveWatchdog for $user (Limited, AtLogOn)."
Write-Host "Verify with:  schtasks /query /tn V4LiveRecorder /xml"
