# Parameters

```
Notes :
✅ : Parameter is optional.
❌ : Parameter is mandatory.
N/A : No default value is set.
```

## Common

| Parameter | Optional | Default Value | Description | Example |
|---------|--------|--------------|-------------|---------|
| `--help` | ❌ | N/A | Help screen. | `--help` |
| `--level_log` | ✅ | N/A | Set level of log. | `--level_log=<debug|info|error|warning>` |
| `--log_file` | ✅ | N/A | Set the log file path. | `--log_file=PATH` |
| `--log_interval_hash` | ✅ | 10000 | Set the time interval (in milliseconds) between logs of information about the hashrate. | `--log_interval_hash=10000` |
| `--host` | ❌ | N/A | Hostname of the pool. | `--host="ethw.2miners.com"` |
| `--port` | ❌ | N/A | Port of the pool. | `--port=2020` |
| `--wallet` | ❌ | N/A | Wallet address. | `--wallet="WALLET"` |
| `--algo` | ❌ | N/A | Algorithm. | `--algo="ethash"` |
| `--workername` | ❌ | N/A | Name of the rig. | `--workername="MyWorkerName"` |
| `--password` | ✅ | N/A | Account password. | `--password="MyPassword"` |
| `--ssl` | ✅ | false | Enable or not the SSL. | `--ssl=<true|false>` |
| `--stale` | ✅ | false | Enable stale share. | `--stale=<true|false>` |
| `--nvidia` | ✅ | true | Enable or disable device nvidia. | `--nvidia=<true|false>` |
| `--amd` | ✅ | true | Enable or disable device amd. | `--amd=<true|false>` |
| `--cpu` | ✅ | false | Enable or disable device cpu. | `--cpu=<true|false>` |
| `--socks5` | ✅ | false | Enable pool connection through a SOCKS5 proxy server on localhost. | `--socks5=<true|false>` |
| `--socks_port` | ✅ | 9050 | The port of the SOCKS5 proxy server on localhost. | `--socks_port=9050` |
| `--api_port` | ✅ | 8080 | Miner API port. | `--api_port=8080` |

## AMD Settings

| Parameter | Optional | Default Value | Description | Example |
|---------|--------|--------------|-------------|---------|
| `--amd_host` | ✅ | N/A | Set default hostname of pool for all GPU AMD. If defined, the parameters `amd_port` and `amd_algo` must be defined. | `--amd_host="ethw.2miners.com"` |
| `--amd_port` | ✅ | N/A | Set port of the pool for all GPU AMD. If defined, the parameters `amd_host` and `amd_algo` must be defined. | `--amd_port=2020` |
| `--amd_algo` | ❌ | N/A | Algorithm. If defined, the parameters `amd_host` and `amd_port` must be defined. | `--amd_algo="ethash"` |

## Nvidia Settings

| Parameter | Optional | Default Value | Description | Example |
|---------|--------|--------------|-------------|---------|
| `--nvidia_host` | ✅ | N/A | Set default hostname of pool for all GPU NVIDIA. If defined, the parameters `nvidia_port` and `nvidia_algo` must be defined. | `--nvidia_host="ethw.2miners.com"` |
| `--nvidia_port` | ✅ | N/A | Set port of the pool for all GPU NVIDIA. If defined, the parameters `nvidia_host` and `nvidia_algo` must be defined. | `--nvidia_port=2020` |
| `--nvidia_algo` | ❌ | N/A | Algorithm. If defined, the parameters `nvidia_host` and `nvidia_port` must be defined. | `--nvidia_algo="ethash"` |

## Specific Device

| Parameter | Optional | Default Value | Description | Example |
|---------|--------|--------------|-------------|---------|
| `--devices_disable` | ✅ | N/A | List device disable. | `--device_disable=0,1` |
| `--device_pool` | ✅ | N/A | Define hostname pool for custom device. | `--device_pool=0:ethw.2miners.com` |
| `--device_port` | ✅ | N/A | Define port for custom device. | `--device_port=0:2020` |
| `--device_password` | ✅ | N/A | Define password for custom device. | `--device_password=0:MyPassword` |
| `--device_algo` | ✅ | N/A | Define algorithm for custom device. | `--device_algo=0:ethash` |
| `--device_wallet` | ✅ | N/A | Define wallet for custom device. | `--device_wallet=0:WALLET` |
| `--device_workername` | ✅ | N/A | Define workername for custom device. | `--device_workername=0:MyWorkerName` |

## RavenMiner

| Parameter | Optional | Default Value | Description | Example |
|---------|--------|--------------|-------------|---------|
| `--rm_rvn_btc` | ✅ | N/A | Mining on ravenminer RVN with BTC wallet | `--rm_rvn_btc=WALLET` |
| `--rm_rvn_eth` | ✅ | N/A | Mining on ravenminer RVN with ETH wallet | `--rm_rvn_eth=WALLET` |
| `--rm_rvn_ltc` | ✅ | N/A | Mining on ravenminer RVN with LTC wallet | `--rm_rvn_ltc=WALLET` |
| `--rm_rvn_bch` | ✅ | N/A | Mining on ravenminer RVN with BCH wallet | `--rm_rvn_bch=WALLET` |
| `--rm_rvn_ada` | ✅ | N/A | Mining on ravenminer RVN with ADA wallet | `--rm_rvn_ada=WALLET` |
| `--rm_rvn_dodge` | ✅ | N/A | Mining on ravenminer RVN with DODGE wallet | `--rm_rvn_dodge=WALLET` |
| `--rm_rvn_matic` | ✅ | N/A | Mining on ravenminer RVN with MATIC wallet | `--rm_rvn_matic=WALLET` |

## Kernel Intensity

| Parameter | Optional | Default Value | Description | Example |
|---------|--------|--------------|-------------|---------|
| `--threads` | ✅ | N/A | Set occupancy threads. | `--threads=128` |
| `--blocks` | ✅ | N/A | Set occupancy blocks. | `--blocks=128` |
| `--occupancy` | ✅ | false | System will define the best occupancy for kernel. | `--occupancy=false` |
| `--internal_loop` | ✅ | 1 | Set internal loop for kernel. | `--internal_loop=1` |

## Smart Mining

| Parameter | Optional | Default Value | Description | Example |
|---------|--------|--------------|-------------|---------|
| `--sm_wallet` | ✅ | N/A | Assign a wallet with a coin. | `--sm_wallet=COIN:WALLET` |
| `--sm_pool` | ✅ | N/A | Assign a pool with a coin. | `--sm_pool=COIN@POOL_URL:POOL_PORT` |
