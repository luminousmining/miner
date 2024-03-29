#!/usr/bin/env bash

# Documentation HiveOS
# https://github.com/minershive/hiveos-linux/blob/master/hive/miners/custom/README.md#h-statssh

stats=$(curl -s "http://127.0.0.1:8080/hiveos/getStats")
total_hash=$(curl -s "http://127.0.0.1:8080/hiveos/getTotalHashrate")
khs=`echo ${total_hash} | jq -r '.total_hash_rate'`


# Debug print
echo "stats => ${stats}"
echo "total_hash => ${total_hash}"
echo "khs => ${khs}"
