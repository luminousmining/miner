# STANDAR MINING

## AUTOLYKOS V2
```bat
miner.exe^
 --host="erg.2miners.com"^
 --port=8888^
 --wallet="YOUR_WALLET"^
 --algo="autolykosv2"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false
```

## PROGPOW QUAI
```bat
miner.exe^
 --stratum="v2"^
 --host="quai.luckypool.io"^
 --port=3333^
 --wallet="YOUR_WALLET"^
 --algo="progpow-quai"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false
```

## FIROPOW
```bat
miner.exe^
 --host="firo.2miners.com"^
 --port=8181^
 --wallet="YOUR_WALLET"^
 --algo="firopow"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false
```

## ETHASH
```bat
miner.exe^
 --host="ethw.2miners.com"^
 --port=2020^
 --wallet="YOUR_WALLET"^
 --algo="ethash"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false
```

## ETCHASH
```bat
miner.exe^
 --host="etc.2miners.com"^
 --port=1010^
 --wallet="YOUR_WALLET"^
 --algo="etchash"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false
```

## KAWPOW
```bat
miner.exe^
 --host="rvn.2miners.com"^
 --port=6060^
 --wallet="YOUR_WALLET"^
 --algo="kawpow"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false
```

## [Optional] Connection via a socks5 proxy
Requires a socks5 proxy server running on localhost. For example, the Linux [tor](https://www.torproject.org/) client runs a socks5 server on the port 9050 by default.
```bat
miner.exe^
 --host="fr.vipor.net"^
 --port=5030^
 --wallet="YOUR_WALLET"^
 --algo="kawpow"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false^
 --socks5=true^
 --socks_port=9050
```

# SPLIT MINING
In these examples the RIG contains more than 2 GPUs !

## Many Coins
All gpu will mine `RVN` on `rvn.2miners.com:6060` except GPUs 1 and 2 !  
The GPU 1 will mine `CLORE` on `clore.2miner.com:2020`.  
The GPU 2 will mine `ETC` on `etc.2miner.com:1010`.
```bat
miner.exe^
 --host="rvn.2miners.com"^
 --port=6060^
 --wallet="YOUR_WALLET"^
 --algo="kawpow"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false^
 --device_algo=1:kawpow^
 --device_pool=1:clore.2miner.com^
 --device_port=1:2020^
 --device_wallet=1:YOUR_WALLET^
 --device_algo=2:etchash^
 --device_pool=2:etc.2miner.com^
 --device_port=2:1010^
 --device_wallet=2:YOUR_WALLET
```

## Many Pools
All gpu will mine `CLORE` on `rvn.2miners.com:6060` except GPUs 1 and 2 !  
The GPU 1 will mine `CLORE` on `stratum-eu.rplant.xyz:17083`.  
The GPU 2 will mine `CLORE` on `pool.eu.woolypooly.com:3126`.
```bat
miner.exe^
 --host="clore.2miners.com"^
 --port=2020^
 --wallet="YOUR_WALLET"^
 --algo="kawpow"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false^
 --device_algo=1:kawpow^
 --device_pool=1:stratum-eu.rplant.xyz^
 --device_port=1:17083^
 --device_wallet=1:YOUR_WALLET^
 --device_algo=2:kawpow^
 --device_pool=2:pool.eu.woolypooly.com^
 --device_port=2:3126^
 --device_wallet=2:YOUR_WALLET
```

## Many Coins And Pools
All gpu will mine `CLORE` on `rvn.2miners.com:6060` except GPUs 1 and 2 !  
The GPU 1 will mine `ERGO` on `de.ergo.herominers.com:1180`.  
The GPU 2 will mine `ETHW` on `pool.eu.woolypooly.com:3096`.
```bat
miner.exe^
 --host="clore.2miners.com"^
 --port=2020^
 --wallet="YOUR_WALLET"^
 --algo="kawpow"^
 --workername="luminousminer"^
 --password="x"^
 --ssl=false^
 --device_algo=1:autolykosv2^
 --device_pool=1:de.ergo.herominers.com^
 --device_port=1:1180^
 --device_wallet=1:YOUR_WALLET^
 --device_algo=2:ethash^
 --device_pool=2:pool.eu.woolypooly.com^
 --device_port=2:3096^
 --device_wallet=2:YOUR_WALLET
```
