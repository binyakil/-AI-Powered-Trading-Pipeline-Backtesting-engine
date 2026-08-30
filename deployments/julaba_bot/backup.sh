#!/bin/bash
# Julaba Trading Bot - Automated Backup Script
# Backs up configuration, models, data, and logs

set -e

# Configuration
JULABA_DIR="/home/opc/julaba"
BACKUP_ROOT="/home/opc/julaba_backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$DATE"
MAX_BACKUPS=30  # Keep last 30 backups

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Julaba Backup Script${NC}"
echo -e "${BLUE}========================${NC}"
echo ""

# Create backup directories
mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}✓${NC} Created backup directory: $BACKUP_DIR"

# Backup JSON configuration and history files
echo -e "\n${BLUE}📄 Backing up JSON files...${NC}"
find "$JULABA_DIR" -maxdepth 1 -name "*.json" -type f -exec cp {} "$BACKUP_DIR/" \;
JSON_COUNT=$(ls "$BACKUP_DIR"/*.json 2>/dev/null | wc -l)
echo -e "${GREEN}✓${NC} Backed up $JSON_COUNT JSON files"

# Backup ML models
if [ -d "$JULABA_DIR/models" ]; then
    echo -e "\n${BLUE}🧠 Backing up ML models...${NC}"
    cp -r "$JULABA_DIR/models" "$BACKUP_DIR/"
    MODEL_COUNT=$(find "$BACKUP_DIR/models" -type f | wc -l)
    echo -e "${GREEN}✓${NC} Backed up $MODEL_COUNT model files"
fi

# Backup historical data
if [ -d "$JULABA_DIR/historical_data" ]; then
    echo -e "\n${BLUE}📊 Backing up historical data...${NC}"
    cp -r "$JULABA_DIR/historical_data" "$BACKUP_DIR/"
    DATA_SIZE=$(du -sh "$BACKUP_DIR/historical_data" | cut -f1)
    echo -e "${GREEN}✓${NC} Backed up historical data ($DATA_SIZE)"
fi

# Backup Python pickle files (ML classifiers)
echo -e "\n${BLUE}🔧 Backing up pickle files...${NC}"
find "$JULABA_DIR" -maxdepth 1 -name "*.pkl" -type f -exec cp {} "$BACKUP_DIR/" \; 2>/dev/null || true
PKL_COUNT=$(ls "$BACKUP_DIR"/*.pkl 2>/dev/null | wc -l)
echo -e "${GREEN}✓${NC} Backed up $PKL_COUNT pickle files"

# Backup important logs (last 1000 lines only to save space)
echo -e "\n${BLUE}📝 Backing up recent logs...${NC}"
if [ -f "$JULABA_DIR/julaba.log" ]; then
    tail -1000 "$JULABA_DIR/julaba.log" > "$BACKUP_DIR/julaba.log.recent"
    echo -e "${GREEN}✓${NC} Backed up recent log entries"
fi

# Create compressed archive
echo -e "\n${BLUE}🗜️  Compressing backup...${NC}"
cd "$BACKUP_ROOT"
tar -czf "julaba_${DATE}.tar.gz" "$DATE" 2>/dev/null
ARCHIVE_SIZE=$(du -sh "julaba_${DATE}.tar.gz" | cut -f1)
echo -e "${GREEN}✓${NC} Created archive: julaba_${DATE}.tar.gz ($ARCHIVE_SIZE)"

# Remove uncompressed backup directory
rm -rf "$DATE"

# Cleanup old backups (keep last MAX_BACKUPS)
echo -e "\n${BLUE}🧹 Cleaning up old backups...${NC}"
BACKUP_COUNT=$(ls -1 "$BACKUP_ROOT"/julaba_*.tar.gz 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt "$MAX_BACKUPS" ]; then
    OLD_COUNT=$((BACKUP_COUNT - MAX_BACKUPS))
    ls -1t "$BACKUP_ROOT"/julaba_*.tar.gz | tail -n +$((MAX_BACKUPS + 1)) | xargs rm -f
    echo -e "${YELLOW}⚠${NC}  Removed $OLD_COUNT old backup(s) (keeping last $MAX_BACKUPS)"
else
    echo -e "${GREEN}✓${NC} Keeping all $BACKUP_COUNT backups (max: $MAX_BACKUPS)"
fi

# Summary
echo -e "\n${BLUE}========================${NC}"
echo -e "${GREEN}✅ Backup completed successfully!${NC}"
echo -e "${BLUE}========================${NC}"
echo ""
echo "📦 Backup location: $BACKUP_ROOT/julaba_${DATE}.tar.gz"
echo "💾 Backup size: $ARCHIVE_SIZE"
echo "📊 Total backups: $(ls -1 "$BACKUP_ROOT"/julaba_*.tar.gz 2>/dev/null | wc -l)"
echo ""

# Optional: Sync to remote server (uncomment and configure)
# echo -e "${BLUE}☁️  Syncing to remote backup...${NC}"
# rsync -az --delete "$BACKUP_ROOT/" user@backup-server:/backups/julaba/
# echo -e "${GREEN}✓${NC} Remote sync completed"

exit 0
