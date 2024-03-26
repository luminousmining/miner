# LuminousMiner

Come talk with us [Discord](https://discord.gg/uTNRBFVz)  
  
Free mining software with 0% fees! Enjoy your mining time.  
  
Project aims for learning and sharing about mining software.  
All algorithms used are referenced.  
  
Different profiles are available:
* STANDAR
* SMART MINING

`STANDAR` we must set your mining session and get 0% fees.  
`SMART MINING` define the list coin coins need and let the system mine for you. 1% fees will be applied.  


Algorithms:
- autolykosv2
- ethash
- etchash
- progpow
- progpowz
- kawpow
- firopow
- evrprogpow

___Common options:___
```
--help            Help screen.
--level_log       [OPTIONAL] Set level of log.
               --level_log=<debug|info|error|warning>
--host             [MANDATORY] Hostname of the pool.
               --host="ethw.2miners.com"
--port             [MANDATORY] Port of the pool.
               --port=2020
--wallet           [MANDATORY] Wallet address.
               -wallet="WALLET"
--algo             [MANDATORY] <ethash>
               --algo="ethash"
--workername       [MANDATORY] Name of the rig.
               --workername="MyWorkerName"
--password         [OPTIONAL] Account password.
               --password="MyPassword"
--ssl              [OPTIONAL] Enable or not the SSL.
               Default value is false.
               --ssl=<true|false>.
--stale            [OPTIONAL] Enable stale share.
               Default value is false.
               --stale=<true|false>
--nvidia           [OPTIONAL] Enable or disable device nvidia.
               Default value is true.
               --nvidia=<true|false>
--amd              [OPTIONAL] Enable or disable device amd.
               Default value is true.
               --amd=<true|false>
--cpu              [OPTIONAL] Enable or disable device cpu.
               Default value is false.
               --cpu=<true|false>
```

___Specific device options:___
```
--devices_disable       [OPTIONAL] List device disable.
                    --device_disable=0,1
--device_pool           [OPTIONAL] Define hostname pool for custom device.
                    --device_pool=0:ethw.2miners.com
--device_port           [OPTIONAL] Define port for custom device.
                    --device_pool=0:2020
--device_password       [OPTIONAL] Define password for custom device.
                    --device_password=0:MyPassword
--device_algo           [OPTIONAL] Define algorithm for custom device.
                    --device_pool=0:ethash
--device_wallet         [OPTIONAL] Define wallet for custom device.
                    --device_pool=0:WALLET
--device_workername     [OPTIONAL] Define workername for custom device.
                    --device_workername=0:MyWorkerName
```

___Smart Mining options:___
```
--sm_wallet arg         [OPTIONAL] assign a wallet with a coin.
                    --sm_wallet=COIN:WALLET
--sm_pool arg           [OPTIONAL] assign a pool with a coin.
                    --sm_pool=COIN@POOL_URL:POOL_PORT
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
[progminer](https://github.com/2miners/progminer)  
[evrprogpowminer](https://github.com/EvrmoreOrg/evrprogpowminer)  
[progpowz](https://github.com/hyle-team/progminer)  

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

## SAST Tools
[PVS-Studio](https://pvs-studio.com/pvs-studio/?utm_source=website&utm_medium=github&utm_campaign=open_source) - static analyzer for C, C++, C#, and Java code.
