# Live watchdog wrapper: restart on any exit (crash or clean).
$repo = "D:\Work\Attempts to Code\BTC Strategy\BTC Strategy"
$python = "C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"
$log = "$repo\v4_system\data\live\_watchdog_console.log"
Set-Location $repo
while ($true) {
    "=== watchdog start $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 $log
    & $python -u -m v4_system.data_engine.live_watchdog *>> $log
    "=== watchdog exit $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 $log
    Start-Sleep -Seconds 5
}
