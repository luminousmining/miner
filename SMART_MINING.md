# Smart Mining

## What is it ?

SmartMining is a protocol allowing you to switch coins during your mining session.
The server will send the best algorithm and associated jobs based on the listed coins.

## Protocol
```
miner                               pool
|                                   |
| --> mining.subscribe              |
| <--- response                     |
| <--- smart_mining.set_algo        |
| <--- smart_mining.set_extra_nonce |
|                                   |
```

`mining.subscribe`:
```json
{
    "id": 1,
    "method": "mining.subscribe",
    "params":
    [
        [
            "COIN_TAG",
            "POOL_HOST",
            POOL_PORT,
            "YOUR_WALLET"
        ],
        ...
    ]
}
```
`COIN_TAG` -> `string`: name of coin, examples CLORE, FIRO,...  
`POOL_HOST` -> `string`: pool host, examples "clore.2miners.com", "pool.eu.woolypooly.com",...  
`POOL_PORT` -> `integer`: pool port, examples 2020,3126,...  
`YOUR_WALLET` -> `string`: your wallet linked to `COIN_TAG`.  
  

`smart_mining.set_algo`:
```json
{
    "id": 1,
    "method": "smart_mining.set_algo",
    "params": "ALGORITHM"
}
```
`ALGORITHM` -> name of algorithm, example kawpow, ethash,...  
  
  
`smart_mining.set_extra_nonce`:
```json
{
    "id": 2,
    "method": "smart_mining.set_extra_nonce",
    "params": PARAMS_EXTRA_NONCE
}
```
`PARAMS_EXTRA_NONCE` -> array or integer  
  