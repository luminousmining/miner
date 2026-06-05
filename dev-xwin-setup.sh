#!/usr/bin/env bash
# Throwaway dev-container setup for iterating the MinGW cross build.
set -e
export DEBIAN_FRONTEND=noninteractive

apt-get update >/dev/null
apt-get install -y --no-install-recommends \
    ca-certificates curl zip unzip tar xz-utils git pkg-config \
    build-essential ninja-build python3 >/dev/null

curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz \
    | tar xz -C /opt
ln -sf /opt/cmake-3.30.5-linux-x86_64/bin/* /usr/local/bin/

url="$(curl -fsSL https://api.github.com/repos/mstorsjo/llvm-mingw/releases/latest \
    | grep -oP '"browser_download_url":\s*"\K[^"]*ucrt-ubuntu-[0-9.]+-x86_64\.tar\.xz' | head -1)"
echo "llvm-mingw: $url"
curl -fsSL "$url" -o /tmp/llvm.tar.xz
mkdir -p /opt/llvm-mingw
tar xJf /tmp/llvm.tar.xz -C /opt/llvm-mingw --strip-components=1
rm -f /tmp/llvm.tar.xz

git clone --depth 1 https://github.com/microsoft/vcpkg /opt/vcpkg >/dev/null 2>&1
/opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics >/dev/null

export PATH=/opt/llvm-mingw/bin:/opt/vcpkg:$PATH
export VCPKG_ROOT=/opt/vcpkg
x86_64-w64-mingw32-clang++ --version | head -1

cd /src
cmake --preset windows-amd-cross
echo CONFIGURE_DONE
