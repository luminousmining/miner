# Requirements
  
## Libraires
- cuda 12.5 => Windows
- cuda 12.4 => Ubuntu
- OpenSSL 1.1.1
- boost 1.85.0
- OpenCL 3.0.15
  
### Windows
- Visual Studio 2022
- Windows SDK 10.0.22621.0
- CMake >= 3.22.4

### Install
cmake : https://github.com/Kitware/CMake/releases/tag/v3.22.4  
cuda : https://developer.nvidia.com/cuda-12-5-0-download-archive  
boost : https://boostorg.jfrog.io/artifactory/main/release/1.85.0/source/boost_1_85_0.zip  
```bat
bootstrap.bat
b2.exe release
b2.exe debug
b2.exe install --prefix=C:\\Boost
```
opencl :  https://github.com/KhronosGroup/OpenCL-SDK.git 
```bat
git fetch --all
git checkout tags/v2023.04.17
git submodule init
git submodule update
if not exist build_opencl mkdir build_opencl
cd build_opencl
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DOPENCL_SDK_BUILD_SAMPLES=ON -DOPENCL_SDK_TEST_SAMPLES=OFF -DCMAKE_INSTALL_PREFIX=C:/OpenCL
cmake --build . --target install
cd ..
```
openssl : https://github.com/openssl/openssl.git  
Need Perl :https://github.com/openssl/openssl/blob/master/NOTES-PERL.md 
```bat
git fetch --all
git checkout tags/OpenSSL_1_1_1t
perl Configure VC-WIN64A
```
Open `Visual Studio Developer Command Promp x86_x64` with privileges !!!
```bat
nmake
nmake install
```
  
### Linux
- clang++ == 10
- CMake >= 3.22

### Install
cmake: https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh  
```sh
wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh --no-check-certificate
sudo mv cmake-3.26.4-linux-x86_64.sh /opt/cmake-3.26.4-linux-x86_64.sh
sudo chmod +x /opt/cmake-3.26.4-linux-x86_64.sh
sudo /opt/cmake-3.26.4-linux-x86_64.sh
sudo cp -r cmake-3.26.4-linux-x86_64 /opt/
sudo rm -rf cmake-3.26.4-linux-x86_64
sudo ln -s /opt/cmake-3.26.4-linux-x86_64/bin/* /usr/local/bin
```
compiler : apt install  
```bah
sudo apt install -y build-essential libstdc++-10 -dev gnutls-dev cppcheck checkinstall clang-10 libx11-dev
```
openssl : https://github.com/openssl/openssl
```sh
git clone https://github.com/openssl/openssl.git
cd openssl
./Configure
make
sudo make install
```
opencl: https://github.com/KhronosGroup/OpenCL-SDK  
```sh
git clone https://github.com/KhronosGroup/OpenCL-SDK.git
git fetch --all
git checkout tags/v2024.05.08
git submodule init
git submodule update
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build . --target install
```
cuda : https://developer.nvidia.com/cuda-12-4-0-download-archive  
```sh
wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```
boost : https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz  
```sh
wget --no-check-certificate https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz
tar -xvf boost_1_85_0.tar.gz
cd boost_1_85_0
./bootstrap.sh --prefix=/usr/local
./b2 release
./b2 debug
sudo ./b2 install
```
# Platforms
  
## Windows

```sh
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
  
## Linux
```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
