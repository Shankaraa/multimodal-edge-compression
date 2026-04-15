param(
    [string]$Distro = "Ubuntu-22.04",
    [string]$VenvPath = "~/.venvs/voxtral-baseline",
    [string]$ModelPath = "models/voxtral-realtime",
    [string]$ConfigPath = "configs/vllm/bf16.yaml",
    [int]$Port = 8080,
    [switch]$DryRun
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$drive = $projectRoot.Substring(0, 1).ToLower()
$rest = $projectRoot.Substring(2).Replace("\", "/")
$wslProjectRoot = "/mnt/$drive$rest"

$bashCommand = @"
source $VenvPath/bin/activate
cd $wslProjectRoot
python scripts/serve_model.py $ModelPath --config $ConfigPath --port $Port
"@

Write-Host "Launching BF16 baseline in WSL..."
Write-Host "Distro: $Distro"
Write-Host "Project: $projectRoot"
Write-Host "WSL path: $wslProjectRoot"

if ($DryRun) {
    Write-Host "Bash command:"
    Write-Host $bashCommand
    exit 0
}

wsl -d $Distro -- bash -lc $bashCommand
