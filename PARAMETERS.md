# Parameters

## Common
```
--help            Help screen.
--level_log       [OPTIONAL] Set level of log.
               --level_log=<debug|info|error|warning>
--log_file        [OPTIONAL] Set level of log.
               --log_file=PATH
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

## AMD Settings
```
--amd_host arg          [OPTIONAL] Set defaut hostname of pool for all gpu 
                        AMD.
                        If defined, the parameters amd_port and amd_algo must
                        be defined.
                    --amd_host="ethw.2miners.com"
--amd_port arg          [OPTIONAL] Set port of the pool for all gpu AMD.
                        If defined, the parameters amd_host and amd_algo must
                        be defined.
                    --amd_port=2020
--amd_algo arg          [MANDATORY] <ethash>
                        If defined, the parameters amd_host and amd_port must
                        be defined.
                        --amd_algo="ethash"
```

## Nvidia Settings
```
--nvidia_host arg       [OPTIONAL] Set defaut hostname of pool for all gpu 
                        NVIDIA.
                        If defined, the parameters nvidia_port and 
                        nvidia_algo must be defined.
                    --nvidia_host="ethw.2miners.com"
--nvidia_port arg       [OPTIONAL] Set port of the pool for all gpu NVIDIA.
                        If defined, the parameters nvidia_host and 
                        nvidia_algo must be defined.
                    --nvidia_port=2020
--nvidia_algo arg       [MANDATORY] <ethash>
                        If defined, the parameters nvidia_host and 
                        nvidia_port must be defined.
                    --nvidia_algo="ethash"
```

## Specific Device
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

## Smart Mining
```
--sm_wallet arg         [OPTIONAL] assign a wallet with a coin.
                    --sm_wallet=COIN:WALLET
--sm_pool arg           [OPTIONAL] assign a pool with a coin.
                    --sm_pool=COIN@POOL_URL:POOL_PORT
```
