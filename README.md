# LuminousMiner

## Requis
- cuda 12.3 => Windows
- cuda 12.0 => Ubuntu
- OpenSSL 1.1.1
- boost 1.83.0
- OpenCL 3.0.15
- Visual Studio 2022

## BUILD
```sh
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## References - Miners
[ethminer](https://github.com/ethereum-mining/ethminer)
[kawpowminer](https://github.com/RavenCommunity/kawpowminer)
[alephium](https://github.com/alephium/gpu-miner)
[cpuminer](https://github.com/pooler/cpuminer)
[bitcoin](https://github.com/pakheili/sha-256-hash-algorithm-bitcoin-miner)
[powkit](https://github.com/sencha-dev/powkit)

## References - Algos
https://en.wikipedia.org/wiki/BLAKE_(hash_function)
https://en.wikipedia.org/wiki/Equihash
https://ergoplatform.org/en/blog/Ergo-and-the-Autolykos-Consensus-Mechanism-Part-I/
https://ergoplatform.org/en/blog/Ergo-and-The-Autolykos-Consensus-Mechanism-Part-II/

## References - Papers
https://pure.manchester.ac.uk/ws/files/85753741/paper.pdf
https://www.mdpi.com/2410-387X/7/4/60
https://www.researchgate.net/publication/255971534_Parallel_Cloud_Computing_Exploiting_Parallelism_on_Keccak_FPGA_and_GPU_Comparison
https://ieeexplore.ieee.org/document/8391706
