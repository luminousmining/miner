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

## Specific GPU
```
--amd_disable       [OPTIONAL] 
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
