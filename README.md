# LuminousMiner
<img src="https://raw.githubusercontent.com/isocpp/logos/master/cpp_logo.png" width="40" height="40"> <img src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Windows_logo_-_2012.png" width="40" height="40">
  
[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com)  
[![discord](https://img.shields.io/discord/1174669427032199188?logo=discord&logoColor=white&label=Chat&color=7289da)](https://discord.gg/F9y3rxBtGP)  
![twitter](https://img.shields.io/twitter/follow/luminousmining)  
  
Welcome on luminousminer, GPU mining open source and free usage!  
Free GPU mining with 0% fees! Enjoy your mining time.  
Project aims for learning and sharing about mining software.  
  
See the [roadmap](https://github.com/luminousmining/miner/tree/main/ROADMAP.md) of luminousminer.  
See the [pools](https://github.com/luminousmining/miner/tree/main/documentation/POOLS.md) tested!  
See the [hiveos](https://github.com/luminousmining/miner/tree/main/documentation/HIVEOS.md) to install on HiveOS!  
  
Different profiles are available:
* STANDAR
* SMART MINING
  
`STANDAR` you must set your mining session (host/algo/...) and get 0% fees.  
`SMART MINING` define the list coin coins need and let the system mine for you.  
## Algorithms
- ethash
- etchash
- progpow
- progpowz
- kawpow
- firopow
- meowpow
- evrprogpow
- progpow_quai

## Internal Documentation
## Build
From scratch (native toolchain):  
[Linux](https://github.com/luminousmining/miner/tree/main/documentation/build/BUILD_LINUX.md)  
[Windows](https://github.com/luminousmining/miner/tree/main/documentation/build/BUILD_WINDOWS.md)  

Docker toolchain (no local toolchain):  
[Docker Linux](https://github.com/luminousmining/miner/tree/main/documentation/build/BUILD_DOCKER_LINUX.md)  
[Docker Windows](https://github.com/luminousmining/miner/tree/main/documentation/build/BUILD_DOCKER_WINDOWS.md)  
[Docker macOS](https://github.com/luminousmining/miner/tree/main/documentation/build/BUILD_DOCKER_MACOS.md)  
[Docker Linux ARM64](https://github.com/luminousmining/miner/tree/main/documentation/build/BUILD_DOCKER_LINUX_ARM64.md)  

[Parameters](https://github.com/luminousmining/miner/tree/main/documentation/PARAMETERS.md)  
[Examples](https://github.com/luminousmining/miner/tree/main/documentation/EXAMPLES.md)  
[Benchmark](https://github.com/luminousmining/miner/tree/main/documentation/BENCHMARK.md)  
[Architecture](https://github.com/luminousmining/miner/tree/main/documentation/ARCHITECTURE.md)  
[Add Algorithm](https://github.com/luminousmining/miner/tree/main/documentation/ADD_ALGORITHM.md)  
[Smart Mining](https://github.com/luminousmining/miner/tree/main/documentation/SMART_MINING.md)  
[Coding Style](https://github.com/luminousmining/miner/tree/main/documentation/CODING_STYLE.md)  

## References - Miners
[ethminer](https://github.com/ethereum-mining/ethminer)  
[kawpowminer](https://github.com/RavenCommunity/kawpowminer)  
[alephium](https://github.com/alephium/gpu-miner)  
[cpuminer](https://github.com/pooler/cpuminer)  
[bitcoin](https://github.com/pakheili/sha-256-hash-algorithm-bitcoin-miner)  
[powkit](https://github.com/sencha-dev/powkit)  
[progminer](https://github.com/2miners/progminer)  
[evrprogpowminer](https://github.com/EvrmoreOrg/evrprogpowminer)  
[progpowz](https://github.com/hyle-team/progminer)  
[firominer](https://github.com/firoorg/firominer)  
[meowpowminer](https://github.com/Meowcoin-Foundation/meowpowminer)  
[quai-gpu-miner](https://github.com/dominant-strategies/quai-gpu-miner)  

## References - Algos
https://en.wikipedia.org/wiki/BLAKE_(hash_function)  
https://en.wikipedia.org/wiki/Equihash  
https://ergoplatform.org/en/blog/Ergo-and-the-Autolykos-Consensus-Mechanism-Part-I/  
https://ergoplatform.org/en/blog/Ergo-and-The-Autolykos-Consensus-Mechanism-Part-II/  
https://docs.qu.ai/learn/introduction  

## References - Papers
https://pure.manchester.ac.uk/ws/files/85753741/paper.pdf  
https://www.mdpi.com/2410-387X/7/4/60  
https://www.researchgate.net/publication/255971534_Parallel_Cloud_Computing_Exploiting_Parallelism_on_Keccak_FPGA_and_GPU_Comparison  
https://ieeexplore.ieee.org/document/8391706  

## References - Libraries
[asio-socks45-client](https://github.com/sehe/asio-socks45-client) - We recognize [Seth Heeren](https://github.com/sehe) for providing the SOCKS5 client implementation.
[BLAKE3](https://github.com/BLAKE3-team/BLAKE3) - We recognize the [BLAKE3 team](https://github.com/BLAKE3-team) (Jack O'Connor, Jean-Philippe Aumasson, Samuel Neves, Zooko Wilcox-O'Hearn) for the official C reference implementation, vendored as the host oracle for the BLAKE3 kernel tests.

## SAST Tools
[PVS-Studio](https://pvs-studio.com/pvs-studio/?utm_source=website&utm_medium=github&utm_campaign=open_source) - static analyzer for C, C++, C#, and Java code.
