#!/bin/bash
# Application-level GPU Power Measurement using AMD-SMI
# Usage ./power_measure.sh "amd-smi ..." "application ..."

echo "Starting monitoring..."
($1) &
rocpid=$!
echo "Sleeping for a while..."
sleep 10
echo "Launching application..."
($2) &
appid=$!
wait "$appid"
echo "Application exited, sleeping for a while..."
sleep 10
echo "Ending monitoring..."
kill "$rocpid"
echo "Making sure CSV dump is completed, sleeping again..."
sleep 60