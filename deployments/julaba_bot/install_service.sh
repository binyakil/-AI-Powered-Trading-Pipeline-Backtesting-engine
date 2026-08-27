#!/bin/bash
# Julaba Trading Bot - Installation Script
# Installs systemd service for automatic startup and process management

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICE_FILE="$SCRIPT_DIR/julaba.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "🚀 Installing Julaba Trading Bot Service..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run with sudo"
    echo "Usage: sudo ./install_service.sh"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Service file not found: $SERVICE_FILE"
    exit 1
fi

# Stop existing service if running
if systemctl is-active --quiet julaba.service; then
    echo "⏸️  Stopping existing service..."
    systemctl stop julaba.service
fi

# Copy service file to systemd
echo "📝 Installing service file..."
cp "$SERVICE_FILE" "$SYSTEMD_DIR/julaba.service"
chmod 644 "$SYSTEMD_DIR/julaba.service"

# Reload systemd
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# Enable service to start on boot
echo "✅ Enabling service..."
systemctl enable julaba.service

# Start service
echo "▶️  Starting service..."
systemctl start julaba.service

# Wait a moment for startup
sleep 3

# Check status
echo ""
echo "📊 Service Status:"
systemctl status julaba.service --no-pager -l || true

echo ""
echo "✅ Installation complete!"
echo ""
echo "📚 Useful commands:"
echo "   sudo systemctl start julaba      # Start the bot"
echo "   sudo systemctl stop julaba       # Stop the bot"
echo "   sudo systemctl restart julaba    # Restart the bot"
echo "   sudo systemctl status julaba     # Check status"
echo "   sudo journalctl -u julaba -f     # View live logs"
echo "   sudo journalctl -u julaba -n 100 # View last 100 log lines"
echo ""
echo "🌐 Dashboard: http://localhost:5000"
