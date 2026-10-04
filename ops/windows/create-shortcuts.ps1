<#
.SYNOPSIS
    Creates two desktop shortcuts: "NHL Dashboard" and "NHL Setup".

.DESCRIPTION
    NHL Dashboard  runs ops\windows\open-dashboard.bat: starts Docker Desktop
                   if needed, waits for the database, starts the dashboard and
                   opens it in your browser (or just opens the browser when the
                   dashboard is already running).
    NHL Setup      runs ops\windows\setup-all.bat: the one-step setup (setup
                   check, database upgrade, catch-up and today's picks, and
                   the scheduled picks jobs; set NHL_ROLE=all first for
                   the props jobs too). Safe to run again.

    Run it once from the repo folder in PowerShell:
        .\ops\windows\create-shortcuts.ps1
    Running it again replaces the two shortcuts (for example after moving
    the repo). Nothing else is changed.

.PARAMETER RepoPath
    The repo folder. Default: two levels up from this script.

.PARAMETER Desktop
    Where the shortcuts go. Default: your desktop (OneDrive's, if Windows
    keeps the desktop there).
#>
[CmdletBinding()]
param(
    # Resolved below: Windows PowerShell 5.1 leaves $PSScriptRoot empty in
    # an advanced script's parameter defaults
    [string]$RepoPath = '',
    [string]$Desktop = [Environment]::GetFolderPath('Desktop')
)

$ErrorActionPreference = 'Stop'
if (-not $RepoPath) {
    $RepoPath = Join-Path $PSScriptRoot '..\..'
}
$RepoPath = (Resolve-Path -LiteralPath $RepoPath).Path
$windows = Join-Path $RepoPath 'ops\windows'
if (-not (Test-Path (Join-Path $windows 'open-dashboard.bat'))) {
    throw "No ops\windows\open-dashboard.bat under '$RepoPath'. Pass -RepoPath <repo folder>."
}
if (-not (Test-Path $Desktop)) {
    throw "The folder '$Desktop' does not exist. Pass -Desktop <folder>."
}

$shell = New-Object -ComObject WScript.Shell

function New-Shortcut {
    param([string]$Name, [string]$Target, [string]$Icon, [string]$Description)
    $path = Join-Path $Desktop "$Name.lnk"
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $Target
    $link.WorkingDirectory = $RepoPath
    $link.IconLocation = "$Icon,0"
    $link.Description = $Description
    $link.WindowStyle = 1        # a normal window: its messages stay readable
    $link.Save()
    Write-Host "Created $path"
}

New-Shortcut -Name 'NHL Dashboard' `
    -Target (Join-Path $windows 'open-dashboard.bat') `
    -Icon (Join-Path $windows 'nhl-dashboard.ico') `
    -Description 'Open the NHL betting dashboard (starts Docker and the database if needed)'

New-Shortcut -Name 'NHL Setup' `
    -Target (Join-Path $windows 'setup-all.bat') `
    -Icon (Join-Path $windows 'nhl-setup.ico') `
    -Description 'Set up the NHL betting system: setup check, database upgrade, catch-up and picks, scheduled jobs'

Write-Host ''
Write-Host 'Double-click "NHL Dashboard" to open the dashboard. Keep its window open while you use it.'
Write-Host 'Double-click "NHL Setup" once to set this PC up (it is safe to run again).'
