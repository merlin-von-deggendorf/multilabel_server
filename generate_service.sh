#!/usr/bin/env bash
# Simple generator for a systemd service


SERVICE_NAME="flaskserver"
DESCRIPTION=${2:-"Service $SERVICE_NAME"}
CWD=$(pwd)
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=${DESCRIPTION}
After=network.target

[Service]
WorkingDirectory=${CWD}
Type=simple
ExecStart=/usr/bin/python3 ${CWD}/dynamicserver.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

echo "Service '${SERVICE_NAME}' installed and started."
