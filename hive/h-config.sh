#!/usr/bin/env bash

conf=" -a CUSTOM_ALGO -o $CUSTOM_URL -u $CUSTOM_TEMPLATE"
[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME

