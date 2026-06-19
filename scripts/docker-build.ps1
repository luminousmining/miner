#requires -Version 5.1
<#
.SYNOPSIS
    Build LuminousMiner via Docker for a chosen OS and GPU backend, extracting
    binaries into dist/<os>-<gpu>/.

.DESCRIPTION
    Everything builds in LINUX container mode -- Windows binaries are
    cross-compiled (clang-cl + xwin). No Docker engine mode switching, no local
    toolchain. GPU backend (amd/nvidia/both/none) and the CPU resolver (-Cpu) are
    independent axes; pass -Cpu with any -Gpu value to fold the CPU resolver into
    the same binary. -Gpu none -Cpu builds a CPU-only binary.

.PARAMETER Os
    linux | windows-cross | all   ('all' builds both for the chosen -Gpu/-Cpu.)

.PARAMETER Gpu
    amd | nvidia | both | none   (default: both)

.PARAMETER Cpu
    Switch. When present, also build the CPU resolver (BUILD_CPU=ON). Combine with
    any -Gpu value. -Gpu none requires -Cpu (otherwise nothing would be built).

.PARAMETER VcpkgRef
    git ref of vcpkg to pin inside the image. Default: empty, which leaves the
    Dockerfile's pinned VCPKG_REF (kept in lockstep with vcpkg.json's
    builtin-baseline) in force. Override only to test a different vcpkg revision --
    a stale override desyncs vcpkg from the baseline and breaks `vcpkg install`.

.EXAMPLE
    scripts/docker-build.ps1 -Os windows-cross -Gpu both
.EXAMPLE
    scripts/docker-build.ps1 -Os linux -Gpu amd -Cpu
.EXAMPLE
    scripts/docker-build.ps1 -Os linux -Gpu none -Cpu
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('linux', 'windows-cross', 'all')]
    [string] $Os,

    [ValidateSet('amd', 'nvidia', 'both', 'none')]
    [string] $Gpu = 'both',

    [switch] $Cpu,

    [string] $VcpkgRef = ''
)

$ErrorActionPreference = 'Stop'
if ($Gpu -eq 'none' -and -not $Cpu) {
    throw "-Gpu none builds no backend; add -Cpu to build the CPU resolver (nothing to build otherwise)."
}
$repoRoot = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $repoRoot 'dist'

function Assert-LinuxEngine {
    $os = (& docker info --format '{{.OSType}}' 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "Docker is not available. Is Docker Desktop running?" }
    if ($os.Trim() -ne 'linux') {
        throw "Docker is in '$($os.Trim())'-container mode; this build needs LINUX containers. Switch and re-run."
    }
}

function Build-One([string] $osName, [string] $gpu, [bool] $cpu) {
    Assert-LinuxEngine
    $cpuArg = if ($cpu) { 'ON' } else { 'OFF' }
    # Dist dir encodes both axes so GPU-only and GPU+CPU artifacts never collide.
    $name = "$osName-$gpu"
    if ($cpu) { $name += "-cpu" }
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
        '--build-arg'
        "CPU=$cpuArg"
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
    foreach ($o in @('linux', 'windows-cross')) { Build-One $o $Gpu ([bool] $Cpu) }
}
else {
    Build-One $Os $Gpu ([bool] $Cpu)
}
