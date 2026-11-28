# Live recorder wrapper: restart on any exit (crash or clean).
$repo = "D:\Work\Attempts to Code\BTC Strategy\BTC Strategy"
$python = "C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"
$log = "$repo\v4_system\data\live\_recorder_console.log"
Set-Location $repo
while ($true) {
    "=== recorder start $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 $log
    & $python -u -m v4_system.data_engine.live_recorder --flush-seconds 60 *>> $log
    "=== recorder exit $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 $log
    Start-Sleep -Seconds 5
}
