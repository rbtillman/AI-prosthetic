#!/bin/bash

# Boot-time Wi-Fi auto-connect script for Raspberry Pi
# Tries to connect to a primary hotspot SSID first, and uses fallback SSID if primary fails.

# For raspberry pi 4, and specific to use needs for CSUM prosthetic hand team

PRIMARY_SSID="YourHotspotSSID"
PRIMARY_PASS="YourHotspotPassword"
FALLBACK_SSID="YourFallbackSSID"
FALLBACK_PASS="YourFallbackPassword"

# Using /boot so that log can be easily accessed from the usb drive
LOGFILE="/boot/wifi_connect.log"

nmcli dev wifi rescan
nmcli dev wifi connect "$PRIMARY_SSID" password "$PRIMARY_PASS" ifname wlan0
sleep 5
IP=$(hostname -I | awk '{print $1}')

if [[ $IP == 192.168.* ]]; then
  echo "$(date): Connected to $PRIMARY_SSID with IP $IP" >> $LOGFILE
else
  echo "$(date): Failed to connect to $PRIMARY_SSID. Trying fallback..." >> $LOGFILE

  # Try fallback network
  nmcli dev wifi connect "$FALLBACK_SSID" password "$FALLBACK_PASS" ifname wlan0
  sleep 5
  IP=$(hostname -I | awk '{print $1}')

  if [[ $IP == 192.168.* ]]; then
    echo "$(date): Connected to fallback network $FALLBACK_SSID with IP $IP" >> $LOGFILE
  else
    echo "$(date): Failed to connect to fallback network $FALLBACK_SSID." >> $LOGFILE
  fi
fi
