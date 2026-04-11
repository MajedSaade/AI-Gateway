#!/bin/bash

# copy the .servcie file
sudo cp yolo.service /etc/systemd/system/

# reload daemon and restart the service
sudo systemctl daemon-reload
sudo systemctl restart yolo.service
sudo systemctl enable yolo.service

# Check if the service is active
if ! systemctl is-active --quiet yolo.service; then
  echo "❌ yolo.service is not running."
  sudo systemctl status yolo.service --no-pager
  exit 1
fi