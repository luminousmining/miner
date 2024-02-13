# LuminousMiner

Free mining software with 0% fees! Enjoy your mining time.  
Project aims for learning and sharing.  
All algorithms used are referenced.  

Algorithms:
- AUTOLYKOS V2
- ETHASH
- KAWPOW

___Common options:___
```
--help                  Help screen.
--level_log arg         [OPTIONAL] Set level of log.
                    --level_log=<debug|info|error|warning>
--host arg              [MANDATORY] Hostname of the pool.
                    --host="ethw.2miners.com"
--port arg              [MANDATORY] Port of the pool.
                    --port=2020
--wallet arg            [MANDATORY] Wallet address.
                    -wallet="WALLET"
--algo arg              [MANDATORY] <ethash>
                    --algo="ethash"
--workername arg        [MANDATORY] Name of the rig.
                    --workername="MyWorkerName"
--password arg          [OPTIONAL] Account password.
                    --password="MyPassword"
--ssl arg               [OPTIONAL] Enable or not the SSL.
                    Default value is false.
                    --ssl=<true|false>.
--stale arg             [OPTIONAL] Enable stale share.
                    Default value is false.
                    --stale=<true|false>
--nvidia arg            [OPTIONAL] Enable or disable device nvidia.
                    Default value is true.
                    --nvidia=<true|false>
--amd arg               [OPTIONAL] Enable or disable device amd.
                    Default value is true.
                    --amd=<true|false>
--cpu arg               [OPTIONAL] Enable or disable device cpu.
                    Default value is false.
                    --cpu=<true|false>
```

___Specific device options:___
```
--devices_disable arg   [OPTIONAL] List device disable.
                    --device_disable=0,1
--device_pool arg       [OPTIONAL] Define hostname pool for custom device.
                    --device_pool=0:ethw.2miners.com
--device_port arg       [OPTIONAL] Define port for custom device.
                    --device_pool=0:2020
--device_password arg   [OPTIONAL] Define password for custom device.
                    --device_password=0:MyPassword
--device_algo arg       [OPTIONAL] Define algorithm for custom device.
                    --device_pool=0:ethash
--device_wallet arg     [OPTIONAL] Define wallet for custom device.
                    --device_pool=0:WALLET
--device_workername arg [OPTIONAL] Define workername for custom device.
                    --device_workername=0:MyWorkerName
```

## Require
- cuda 12.3 => Windows
- cuda 12.0 => Ubuntu
- OpenSSL 1.1.1
- boost 1.83.0
- OpenCL 3.0.15
- Visual Studio 2022

## Build
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
