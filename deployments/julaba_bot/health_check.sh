#!/bin/bash
# Julaba Trading Bot - Health Check Script
# Monitors bot status and alerts if issues detected

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DASHBOARD_URL="http://localhost:5000/api/data"
MAX_DATA_AGE_SECONDS=600  # Alert if data older than 10 minutes

echo -e "${BLUE}🏥 Julaba Health Check${NC}"
echo -e "${BLUE}=====================${NC}"
echo ""

# Check if bot process is running (systemd or manual)
echo -n "🔍 Checking bot process... "

# Check systemd service first
if systemctl is-active --quiet julaba.service 2>/dev/null; then
    BOT_PID=$(systemctl show -p MainPID --value julaba.service)
    echo -e "${GREEN}✅ Running via systemd (PID: $BOT_PID)${NC}"
    RUNNING_MODE="systemd"
else
    # Check for manual process
    BOT_COUNT=$(ps aux | grep "python3 bot.py" | grep -v grep | wc -l)
    
    if [ "$BOT_COUNT" -eq 0 ]; then
        echo -e "${RED}❌ NOT RUNNING${NC}"
        echo -e "${YELLOW}⚠️  Start with systemd: sudo systemctl start julaba${NC}"
        echo -e "${YELLOW}   Or manually: cd /home/opc/julaba && nohup python3 bot.py --dashboard > /dev/null 2>&1 &${NC}"
        exit 1
    elif [ "$BOT_COUNT" -gt 1 ]; then
        echo -e "${YELLOW}⚠️  MULTIPLE INSTANCES ($BOT_COUNT)${NC}"
        echo -e "${YELLOW}⚠️  Kill duplicates with: pkill -9 -f 'python3 bot.py'${NC}"
        ps aux | grep "python3 bot.py" | grep -v grep
        exit 1
    else
        echo -e "${GREEN}✅ Running manually (PID: $(pgrep -f 'python3 bot.py'))${NC}"
        echo -e "${YELLOW}   Consider: sudo systemctl start julaba (for auto-restart)${NC}"
        RUNNING_MODE="manual"
    fi
fi

# Check dashboard responsiveness
echo -n "🌐 Checking dashboard... "
if curl -s --max-time 5 "$DASHBOARD_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Responsive${NC}"
else
    echo -e "${RED}❌ Not responding${NC}"
    echo -e "${YELLOW}⚠️  Dashboard at port 5000 is not responding${NC}"
    exit 1
fi

# Get detailed status from API
echo ""
echo -e "${BLUE}📊 System Status:${NC}"
DATA=$(curl -s --max-time 5 "$DASHBOARD_URL")

if [ -z "$DATA" ]; then
    echo -e "${RED}❌ Could not retrieve status data${NC}"
    exit 1
fi

# Parse JSON data (requires jq, fallback to grep if not available)
if command -v jq &> /dev/null; then
    SYMBOL=$(echo "$DATA" | jq -r '.symbol // "N/A"')
    BALANCE=$(echo "$DATA" | jq -r '.balance // 0')
    AI_MODE=$(echo "$DATA" | jq -r '.ai_mode // "unknown"')
    HAS_POSITION=$(echo "$DATA" | jq -r '.position // false')
    ML_SAMPLES=$(echo "$DATA" | jq -r '.ml_samples // 0')
    ML_TRAINED=$(echo "$DATA" | jq -r '.ml_trained // false')
    
    echo "   Symbol: $SYMBOL"
    echo "   Balance: \$$BALANCE"
    echo "   AI Mode: $AI_MODE"
    echo "   Position: $([ "$HAS_POSITION" == "true" ] && echo "Yes" || echo "None")"
    echo "   ML Samples: $ML_SAMPLES (Trained: $([ "$ML_TRAINED" == "true" ] && echo "Yes" || echo "No"))"
else
    # Fallback if jq not installed
    echo "   (Install jq for detailed status: sudo yum install jq)"
    echo "$DATA" | grep -o '"symbol":"[^"]*"' | head -1
fi

# Check log file size
echo ""
echo -e "${BLUE}📝 Log Status:${NC}"
LOG_FILE="/home/opc/julaba/julaba.log"
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LOG_LINES=$(wc -l < "$LOG_FILE")
    echo "   Size: $LOG_SIZE"
    echo "   Lines: $LOG_LINES"
    
    # Check for recent errors
    RECENT_ERRORS=$(tail -100 "$LOG_FILE" | grep -c "ERROR" || echo "0")
    if [ "$RECENT_ERRORS" -gt 0 ]; then
        echo -e "   ${YELLOW}⚠️  $RECENT_ERRORS recent errors found${NC}"
        echo "   Run: tail -100 $LOG_FILE | grep ERROR"
    else
        echo -e "   ${GREEN}✅ No recent errors${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  Log file not found${NC}"
fi

# Check disk space
echo ""
echo -e "${BLUE}💾 Disk Space:${NC}"
DISK_USAGE=$(df -h /home/opc/julaba | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo -e "   ${RED}❌ Critical: ${DISK_USAGE}% used${NC}"
elif [ "$DISK_USAGE" -gt 80 ]; then
    echo -e "   ${YELLOW}⚠️  Warning: ${DISK_USAGE}% used${NC}"
else
    echo -e "   ${GREEN}✅ ${DISK_USAGE}% used${NC}"
fi

# Check memory usage
echo ""
echo -e "${BLUE}🧠 Memory Usage:${NC}"
if [ "$RUNNING_MODE" == "systemd" ]; then
    BOT_MEM=$(systemctl show -p MemoryCurrent --value julaba.service | awk '{printf "%.0f", $1/1024/1024}')
    echo "   Bot process: ${BOT_MEM}MB (systemd managed)"
else
    BOT_PID=$(pgrep -f 'python3 bot.py' | head -1)
    if [ -n "$BOT_PID" ]; then
        BOT_MEM=$(ps -o rss= -p $BOT_PID | awk '{printf "%.0f", $1/1024}')
        echo "   Bot process: ${BOT_MEM}MB"
    else
        echo "   N/A"
    fi
fi

echo ""
echo -e "${BLUE}=====================${NC}"
echo -e "${GREEN}✅ Health check complete${NC}"
echo ""
echo "📚 Useful commands:"
if [ "$RUNNING_MODE" == "systemd" ]; then
    echo "   sudo systemctl status julaba         # Check status"
    echo "   sudo systemctl restart julaba        # Restart bot"
    echo "   sudo journalctl -u julaba -f         # Live logs"
    echo "   sudo journalctl -u julaba -n 100     # Last 100 log lines"
else
    echo "   tail -f $LOG_FILE                # Live logs"
    echo "   curl -s $DASHBOARD_URL | jq      # API status"
    echo "   ps aux | grep 'python3 bot.py'   # Process info"
    echo "   sudo systemctl start julaba      # Switch to systemd"
fi
echo ""

exit 0
