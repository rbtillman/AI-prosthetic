# Raspberry Pi Wi-Fi and SSH Setup Guide

---

## Objective
Set up a Raspberry Pi to automatically connect to a primary Wi-Fi hotspot (e.g., Windows Mobile Hotspot), fall back to a secondary Wi-Fi network if needed, and self-heal using a watchdog script. Logs are stored in `/boot` for easy access. SSH is used for remote access and should be configured in rPi imager during OS setup.

Additionally, modify settings for SSH terminal connection. The Pi imager can auto-setup SSH, but a few tips and tricks make the workflow a little smoother.  

These instructions are written for the next team, and I am trying to assume no prior knowlege of the rPi, linux OS, or whatever. 

## Notes:
Reccomend to use a windows mobile hotspot as the primary network for CMA project team. Campus wifi authentication method is a pain.  This makes the process simpler.  

For headless OS, a wifi network should be preconfigured from the rPi imager prior to installation of these services. Then SSH into the pi to configure these, reboot, and reconnect. Be careful if using this method as improper configuration can prevent access to the pi from the network.

For graphical OS, (with a monitor and keyboard for the pi), place the files into /boot on the storage device, then access from there. Or, just use the GUI to setup a network and be lazy (this is probably a little overkill, but doing it all so network and ssh is reliable).  

---

## Folder Contents
Assuming all files are located in a project folder:

```
project-folder/
├── rpi_hotspot_connect.sh            # Main connect script
├── rpi_wifi_watchdog.sh              # Watchdog script
├── rpi-hotspot.service               # Systemd service for boot-time Wi-Fi connect
├── rpi_wifi_watchdog.service         # Systemd service for watchdog
├── rpi_wifi_watchdog.timer           # Systemd timer to run watchdog every 5 minutes
```

---

## Installation Steps (One-Time Setup)
This assumes the scripts are somewhere on the Pi's FS. If not, use FTP, paste into terminal, whatever, just figure it out

### 1. Move Scripts to Proper Locations
```bash
sudo cp rpi_hotspot_connect.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/rpi_hotspot_connect.sh

sudo cp rpi_wifi_watchdog.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/rpi_wifi_watchdog.sh
```

### 2. Install Systemd Service for Hotspot Connection on Boot
```bash
sudo cp rpi-hotspot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rpi-hotspot.service
```
Then start and test it:
```bash
sudo systemctl start rpi-hotspot.service
sudo systemctl status rpi-hotspot.service
```

### 3. Install Watchdog Service and Timer
```bash
sudo cp rpi_wifi_watchdog.service /etc/systemd/system/
sudo cp rpi_wifi_watchdog.timer /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable rpi_wifi_watchdog.timer
sudo systemctl start rpi_wifi_watchdog.timer
```
Check it:
```bash
sudo systemctl list-timers | grep rpi_wifi_watchdog
cat /boot/wifi_watchdog.log
```

---

## Script Configuration

### rpi_hotspot_connect.sh
- Edit and set your SSIDs and passwords:
```bash
PRIMARY_SSID="YourHotspotSSID"
PRIMARY_PASS="YourHotspotPassword"
FALLBACK_SSID="YourFallbackSSID"
FALLBACK_PASS="YourFallbackPassword"
```
- This script:
  - Connects to the primary hotspot
  - Falls back to the secondary if the primary fails
  - Logs results to `/boot/wifi_connect.log`

### rpi_wifi_watchdog.sh
- This script runs every 5 minutes
- Pings the hotspot gateway (default: `192.168.137.1`)
- May need to change this but idk
- If unreachable, re-runs the connect script to reconnect
- Logs results to `/boot/wifi_watchdog.log`

---

## Verifying Setup
- Reboot the Pi:
```bash
sudo reboot
```
- Wait ~2 minutes, then check logs:
```bash
cat /boot/wifi_connect.log
cat /boot/wifi_watchdog.log
```
- Verify connection:
```bash
hostname -I
nmcli -t -f active,ssid dev wifi
```

---

## Notes
- Fallback Wi-Fi network should use DHCP (no static config needed)
- The logs are kept in `/boot` so they can be accessed externally
- All scripts run via systemd, robust and reboot-safe

---

## Optional Enhancements
- Rotate or archive logs to prevent boot partition overflow

## SSH stuff
### From Windows powershell: 
After the pi is setup to connect to your hotspot (or whatever network), you can access it via SSH with:

```bash
ssh user@raspberrypi.local
```
Replace "user" with a username configured on the pi. It may default to pi. I reccomend setting up a user and password from the pi imager before installing the OS

If not connected to the windows hotspot and is on some other network, this may still work, but you also might need the IP, which can be found by logging into the router. 
If this doesnt work while using the hotspot, go to the mobile hotspot page in windows settings and make sure it is connected, and it will list it's IP there.  

Then, access it with:
```bash
ssh user@192.137.420.69
```
(replace the IP with the assigned IP)

This should work from within WSL also if you prefer to do it that way. 