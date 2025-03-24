#!/bin/bash

# Wi-Fi watchdog for Raspberry Pi using fallback hotspot connect script
# If connection to the gateway fails, rerun the hotspot connect script

# Prefer the hotspot (Application specific!!)
PRIMARY_GATEWAY="192.168.137.1"
LOGFILE="/boot/wifi_watchdog.log"
CONNECT_SCRIPT="/usr/local/bin/rpi_hotspot_connect.sh"

# Check if gateway is reachable
ping -c 2 -W 3 $PRIMARY_GATEWAY > /dev/null
if [ $? -eq 0 ]; then
  echo "$(date): Gateway reachable. Connection OK." >> $LOGFILE
  exit 0
else
  echo "$(date): Gateway unreachable. Running reconnect script..." >> $LOGFILE
  $CONNECT_SCRIPT
fi
