# Chatette backend firewall - run in PowerShell **as Administrator**:
#   powershell -ExecutionPolicy Bypass -File C:\chatette\chatette_ai_backend\tools\firewall_chatette.ps1
#
# What it does:
#   1. Disables the two broad "python.exe" rules Windows created when Python was first allowed
#      (any port, from any address). While they are on, no narrower rule restricts anything.
#   2. Allows port 8000 (the Chatette backend) only from the home network and from Tailscale.
#
# Undo:  powershell -ExecutionPolicy Bypass -File ...\firewall_chatette.ps1 -Undo

param([switch]$Undo)

$ErrorActionPreference = "Stop"
$python   = "C:\users\erick\appdata\local\programs\python\python312\python.exe"
$ruleName = "Chatette backend (home network + Tailscale)"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Please run this script as Administrator." -ForegroundColor Red
    exit 1
}

# The broad rules: DisplayName "python.exe", program = this Python, allow, inbound
$broad = Get-NetFirewallRule -Direction Inbound -Action Allow |
    Where-Object { $_.DisplayName -eq "python.exe" -and
                   (($_ | Get-NetFirewallApplicationFilter).Program -eq $python) }

if ($Undo) {
    Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    $broad | Enable-NetFirewallRule
    Write-Host "Undone: Chatette rule removed, python.exe rules enabled again." -ForegroundColor Yellow
    exit 0
}

$broad | Disable-NetFirewallRule
Write-Host "Disabled $(@($broad).Count) broad python.exe rule(s)."

Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $ruleName `
    -Description "Chatette API (port 8000): home network and Tailscale only." `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Program $python `
    -RemoteAddress @("LocalSubnet", "100.64.0.0/10", "fd7a:115c:a1e0::/48") `
    -Profile Private | Out-Null
Write-Host "Added: $ruleName" -ForegroundColor Green
Write-Host "  port 8000 from: home network (LocalSubnet), Tailscale (100.64.0.0/10, fd7a:115c:a1e0::/48)"
Write-Host ""
Write-Host "If Windows later asks whether Python may 'communicate on networks', choose Cancel -"
Write-Host "allowing it would create a new broad rule and undo this."
