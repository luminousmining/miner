#!/usr/bin/env bash

# Documentation HiveOS
# https://github.com/minershive/hiveos-linux/blob/master/hive/miners/custom/README.md#h-runsh

[[ `ps aux | grep "\./miner" | grep -v grep | wc -l` != 0 ]] &&
  echo -e "${RED}miner is already running${NOCOLOR}" &&
  exit 1

CUSTOM_ARGUMENT_USER=`cat /hive/miners/custom/luminousminer/mining_arguments.conf`
[[ -z ${CUSTOM_ARGUMENT_USER} ]] &&
  echo -e "${RED}Command argument are empty.${NOCOLOR}" &&
  exit 1

cd /hive/miners/custom/luminousminer/
./miner ${CUSTOM_ARGUMENT_USER}
