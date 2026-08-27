#!/bin/bash
# Setup automated backups for Julaba
# This adds a cron job to run backups every 6 hours

BACKUP_SCRIPT="/home/opc/julaba/backup.sh"

echo "⏰ Setting up automated backups..."

# Check if backup script exists
if [ ! -f "$BACKUP_SCRIPT" ]; then
    echo "❌ Backup script not found: $BACKUP_SCRIPT"
    exit 1
fi

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$BACKUP_SCRIPT"; then
    echo "⚠️  Cron job already exists!"
    echo "Current crontab:"
    crontab -l | grep julaba
    exit 0
fi

# Add cron job (every 6 hours)
(crontab -l 2>/dev/null; echo "0 */6 * * * $BACKUP_SCRIPT >> /home/opc/julaba_backup.log 2>&1") | crontab -

echo "✅ Automated backup configured!"
echo ""
echo "Schedule: Every 6 hours (0:00, 6:00, 12:00, 18:00)"
echo "Log: /home/opc/julaba_backup.log"
echo ""
echo "To verify:"
echo "  crontab -l"
echo ""
echo "To remove:"
echo "  crontab -e  # Then delete the julaba line"
