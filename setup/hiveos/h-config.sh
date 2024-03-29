#!/usr/bin/env bash

# Documentation HiveOS
# https://github.com/minershive/hiveos-linux/blob/master/hive/miners/custom/README.md#h-configsh

# Check if algo is supported
[[ ${CUSTOM_ALGO} != "progpow" ]] &&
    [[ ${CUSTOM_ALGO} != "progpowz" ]] &&
    [[ ${CUSTOM_ALGO} != "evrprogpow" ]] &&
    [[ ${CUSTOM_ALGO} != "kawpow" ]] &&
    [[ ${CUSTOM_ALGO} != "firopow" ]] &&
    [[ ${CUSTOM_ALGO} != "ethash" ]] &&
    [[ ${CUSTOM_ALGO} != "etchash" ]] &&
    echo "${RED}Algo[${CUSTOM_ALGO}] is not supported !${NOCOLOR}" &&
    exit 1

# Extract host
# Extract port
IFS=':'
read -a strarr <<< "${CUSTOM_URL}"
HOST=${strarr[0]}
PORT=${strarr[1]}

# Extract wallet
# Extract workername
IFS='.'
read -a strarr <<< "${CUSTOM_TEMPLATE}"
WALLET=${strarr[0]}
WORKERNAME=${strarr[1]}

# Define the password
PASSWORD="x"
if [[ ! -z ${CUSTOM_PASS} ]]; then
    PASSWORD=${CUSTOM_PASS}
fi


# Write custom arguments for luminousminer
echo \
" --host=${HOST}"\
" --port=${PORT}"\
" --algo=${CUSTOM_ALGO}"\
" --wallet=${WALLET}"\
" --workername=${WORKERNAME}"\
" --password=${PASSWORD}"\
" --log_file=/var/log/luminousminer.log"\
" ${CUSTOM_USER_CONFIG}"\
> ${MINER_DIR}/luminousminer/mining_arguments.conf
