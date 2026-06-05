#requires -Version 5.1
<#
.SYNOPSIS
    Build LuminousMiner for one or more targets using Docker, extracting the
    resulting binaries into dist/<target>/.

.DESCRIPTION
    Linux targets build in Linux containers and export via BuildKit (`-o`).
    Windows targets build in Windows containers and are extracted with
    `docker create` + `docker cp` (Windows images cannot use a `scratch`
    export stage).

    Docker Desktop runs EITHER Linux OR Windows containers at a time. This
    script checks the current engine mode and tells you to switch when the
    requested target needs the other mode.

.PARAMETER Target
    linux-amd | linux-nvidia | windows-amd | all
    ('all' builds the three supported targets; windows-nvidia is deferred.)

.PARAMETER VcpkgRef
    git ref of vcpkg to pin inside the image (default: master).

.EXAMPLE
    scripts/docker-build.ps1 -Target linux-amd
.EXAMPLE
    scripts/docker-build.ps1 -Target all
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('linux-amd', 'linux-nvidia', 'windows-amd', 'all')]
    [string] $Target,

    [string] $VcpkgRef = 'master'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $repoRoot 'dist'

function Get-EngineOs {
    $os = (& docker info --format '{{.OSType}}' 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "Docker is not available. Is Docker Desktop running?" }
    return $os.Trim()
}

function Assert-EngineMode([string] $required) {
    $current = Get-EngineOs
    if ($current -ne $required) {
        throw @"
Docker is in '$current'-container mode but this target needs '$required' containers.
Switch modes, then re-run:
  * Tray icon -> 'Switch to $required containers...'
  * or: & '$env:ProgramFiles\Docker\Docker\DockerCli.exe' -SwitchDaemon
"@
    }
}

function Build-LinuxTarget([string] $name) {
    Assert-EngineMode 'linux'
    $out = Join-Path $distRoot $name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Host "==> Building $name (Linux container) -> $out" -ForegroundColor Cyan
    $env:DOCKER_BUILDKIT = '1'
    & docker build `
        -f (Join-Path $repoRoot "docker/Dockerfile.$name") `
        --build-arg "VCPKG_REF=$VcpkgRef" `
        --target artifact `
        -o $out `
        $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "docker build failed for $name" }
    Write-Host "==> $name binaries in $out" -ForegroundColor Green
}

function Build-WindowsTarget([string] $name) {
    Assert-EngineMode 'windows'
    $out = Join-Path $distRoot $name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $tag = "luminousminer:$name"
    Write-Host "==> Building $name (Windows container) -> $out" -ForegroundColor Cyan
    $env:DOCKER_BUILDKIT = '1'
    & docker build `
        -f (Join-Path $repoRoot "docker/Dockerfile.$name") `
        --build-arg "VCPKG_REF=$VcpkgRef" `
        --isolation=hyperv `
        -t $tag `
        $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "docker build failed for $name" }

    # Windows images can't export via `scratch`; copy out of a stopped container.
    $cid = (& docker create $tag).Trim()
    try {
        & docker cp "${cid}:C:\src\build\$name\bin\." $out
        if ($LASTEXITCODE -ne 0) { throw "docker cp failed for $name" }
    }
    finally {
        & docker rm $cid | Out-Null
    }
    Write-Host "==> $name binaries in $out" -ForegroundColor Green
}

function Build-Target([string] $name) {
    switch ($name) {
        'linux-amd'    { Build-LinuxTarget   $name }
        'linux-nvidia' { Build-LinuxTarget   $name }
        'windows-amd'  { Build-WindowsTarget $name }
        default        { throw "Unknown target '$name'" }
    }
}

if ($Target -eq 'all') {
    Write-Host "Building all supported targets. Linux targets first; you'll be prompted to switch container mode before the Windows target." -ForegroundColor Yellow
    foreach ($t in @('linux-amd', 'linux-nvidia', 'windows-amd')) {
        try { Build-Target $t }
        catch { Write-Warning "Skipped/failed '$t': $($_.Exception.Message)" }
    }
}
else {
    Build-Target $Target
}
