#requires -Version 5.1
<#
.SYNOPSIS
    Build LuminousMiner via Docker for a chosen OS and GPU backend, extracting
    binaries into dist/<os>-<gpu>/.

.DESCRIPTION
    Everything builds in LINUX container mode -- Windows binaries are
    cross-compiled (clang-cl + xwin). No Docker engine mode switching, no local
    toolchain. Backends: amd (OpenCL), nvidia (CUDA), or both in one binary.

.PARAMETER Os
    linux | windows-cross | all   ('all' builds both for the chosen -Gpu.)

.PARAMETER Gpu
    amd | nvidia | both   (default: both)

.PARAMETER VcpkgRef
    git ref of vcpkg to pin inside the image. Default: empty, which leaves the
    Dockerfile's pinned VCPKG_REF (kept in lockstep with vcpkg.json's
    builtin-baseline) in force. Override only to test a different vcpkg revision --
    a stale override desyncs vcpkg from the baseline and breaks `vcpkg install`.

.EXAMPLE
    scripts/docker-build.ps1 -Os windows-cross -Gpu both
.EXAMPLE
    scripts/docker-build.ps1 -Os all -Gpu amd
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('linux', 'windows-cross', 'all')]
    [string] $Os,

    [ValidateSet('amd', 'nvidia', 'both')]
    [string] $Gpu = 'both',

    [string] $VcpkgRef = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $repoRoot 'dist'

function Assert-LinuxEngine {
    $os = (& docker info --format '{{.OSType}}' 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "Docker is not available. Is Docker Desktop running?" }
    if ($os.Trim() -ne 'linux') {
        throw "Docker is in '$($os.Trim())'-container mode; this build needs LINUX containers. Switch and re-run."
    }
}

function Build-One([string] $osName, [string] $gpu) {
    Assert-LinuxEngine
    $name = "$osName-$gpu"
    $out  = Join-Path $distRoot $name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Host "==> Building $name (Linux container) -> $out" -ForegroundColor Cyan
    $env:DOCKER_BUILDKIT = '1'
    $dockerArgs = @(
        'build'
        '-f'
        (Join-Path $repoRoot "docker/Dockerfile.$osName")
        '--build-arg'
        "GPU=$gpu"
        '--target'
        'artifact'
        '-o'
        $out
    )
    # Only override the Dockerfile's pinned VCPKG_REF when the caller explicitly asks;
    # otherwise the image's ARG (in lockstep with vcpkg.json's builtin-baseline) wins.
    if (-not [string]::IsNullOrWhiteSpace($VcpkgRef)) {
        $dockerArgs += '--build-arg'
        $dockerArgs += "VCPKG_REF=$VcpkgRef"
    }
    $dockerArgs += $repoRoot
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) { throw "docker build failed for $name" }
    Write-Host "==> $name binaries in $out" -ForegroundColor Green
}

if ($Os -eq 'all') {
    foreach ($o in @('linux', 'windows-cross')) { Build-One $o $Gpu }
}
else {
    Build-One $Os $Gpu
}
