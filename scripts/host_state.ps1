<#
    Roadmap v3 §7.4 item 3 — record the Windows-side conditions a timing run was made
    under. WSL cannot see AC/battery or the host power scheme, so this runs on the
    Windows side and `dronevla.profile_env --host-state` merges the JSON into env.json.

    Usage (from Windows PowerShell):
        powershell -ExecutionPolicy Bypass -File scripts\host_state.ps1
        powershell -ExecutionPolicy Bypass -File scripts\host_state.ps1 -OutFile <path>
#>
param(
    # Default: <repo>/reports/host_state.json, resolved from this script's own location so
    # it behaves the same from any working directory, including over a UNC path.
    [string]$OutFile = (Join-Path $PSScriptRoot '..\reports\host_state.json')
)

$scheme = (powercfg /getactivescheme) -join ' '
$schemeGuid = $null
$schemeName = $null
if ($scheme -match 'GUID:\s*([0-9a-fA-F-]+)\s*\(([^)]*)\)') {
    $schemeGuid = $Matches[1]
    $schemeName = $Matches[2]
}

$batteryStatusText = @{
    1 = 'discharging'; 2 = 'on AC'; 3 = 'fully charged'; 4 = 'low'; 5 = 'critical'
    6 = 'charging'; 7 = 'charging and high'; 8 = 'charging and low'
    9 = 'charging and critical'; 10 = 'undefined'; 11 = 'partially charged'
}

$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
$onAc = $null
if ($battery) {
    $code = [int]$battery.BatteryStatus
    $text = $batteryStatusText[$code]
    if ($null -eq $text) { $text = 'unknown' }
    # Code 1 is the only value that means "running off the battery".
    $onAc = ($code -ne 1)
    $batteryInfo = [ordered]@{
        present                  = $true
        battery_status_code      = $code
        battery_status           = $text
        estimated_charge_percent = $battery.EstimatedChargeRemaining
    }
} else {
    $batteryInfo = [ordered]@{ present = $false }
}

$os  = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1

# Background load: heaviest processes by accumulated CPU time, as a coarse record of what
# else was running. This is not a sampled utilisation measurement.
$top = Get-Process | Where-Object { $_.CPU -gt 0 } |
       Sort-Object CPU -Descending | Select-Object -First 8 |
       ForEach-Object { [ordered]@{ name = $_.ProcessName; cpu_seconds = [math]::Round($_.CPU, 1) } }

$state = [ordered]@{
    captured_local            = (Get-Date).ToString('o')
    os_caption                = $os.Caption
    os_version                = $os.Version
    power_scheme_name         = $schemeName
    power_scheme_guid         = $schemeGuid
    on_ac_power               = $onAc
    battery                   = $batteryInfo
    cpu_name                  = $cpu.Name
    cpu_load_percent_instant  = $cpu.LoadPercentage
    host_ram_total_mb         = [math]::Round($os.TotalVisibleMemorySize / 1024)
    host_ram_free_mb          = [math]::Round($os.FreePhysicalMemory / 1024)
    process_count             = (Get-Process).Count
    top_processes_by_cpu_time = $top
    note                      = 'cpu_load_percent_instant is a single WMI sample, not an average. CPU temperature is not exposed by this hardware through WMI.'
}

$json = $state | ConvertTo-Json -Depth 5

$dir = Split-Path -Parent $OutFile
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
# Resolve to a provider path and write without a BOM: json.load() rejects a UTF-8 BOM.
$target = Join-Path (Resolve-Path -LiteralPath $dir).ProviderPath (Split-Path -Leaf $OutFile)
[System.IO.File]::WriteAllText($target, $json, (New-Object System.Text.UTF8Encoding $false))
Write-Output "wrote $target"
