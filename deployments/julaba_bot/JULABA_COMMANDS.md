# JULABA Trading Bot - Commands & Troubleshooting Guide

## 📋 Quick Reference

### Basic Commands

| Command | Description |
|---------|-------------|
| `julaba start` | Start the trading bot |
| `julaba stop` | Stop the trading bot |
| `julaba restart` | Restart the trading bot |
| `julaba status` | Show bot status and positions |
| `julaba help` | Show all available commands |

### Monitoring Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `julaba logs` | `julaba l` | View live logs (all) |
| `julaba logs ai` | | View AI decision logs only |
| `julaba logs trade` | | View trade entry/exit logs |
| `julaba logs error` | | View error logs only |
| `julaba health` | | Run health check |

### Trading Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `julaba positions` | `julaba pos` | Show all open positions |
| `julaba balance` | `julaba bal` | Show current balance |
| `julaba trades` | | Show last 10 trades |
| `julaba trades 20` | | Show last 20 trades |
| `julaba performance` | `julaba perf` | Performance summary |
| `julaba close ETHUSDT` | | Close specific position |
| `julaba close` | | Interactive position close |

### System Commands

| Command | Description |
|---------|-------------|
| `julaba config` | Show current configuration |
| `julaba dashboard` | Open dashboard URL |

---

## 🔧 Troubleshooting Guide

### Problem: Bot Won't Start - Lock File Exists

**Symptoms:**
- Bot says "already running" but it's not
- Lock file blocking startup

**Solution:**
```bash
# Remove the lock file
rm -f /home/opc/julaba/julaba.lock

# Then start normally
julaba start
```

**One-liner fix:**
```bash
rm -f /home/opc/julaba/julaba.lock && julaba start
```

---

### Problem: Multiple Bot Instances Running

**Symptoms:**
- High CPU usage
- Duplicate trades
- Conflicting operations

**Check for multiple instances:**
```bash
# List all bot processes
pgrep -af "python3 bot.py"

# Count instances
pgrep -c "python3 bot.py"
```

**Kill ALL instances:**
```bash
# Graceful kill
pkill -f "python3 bot.py"

# Force kill if graceful doesn't work
pkill -9 -f "python3 bot.py"

# Clean up and restart
rm -f /home/opc/julaba/julaba.lock
julaba start
```

**Complete cleanup one-liner:**
```bash
pkill -9 -f "python3 bot.py"; rm -f /home/opc/julaba/julaba.lock; sleep 2; julaba start
```

---

### Problem: Dashboard Not Responding

**Symptoms:**
- Dashboard shows loading forever
- API calls timeout
- Port 5000 not responding

**Check if dashboard is running:**
```bash
# Check if port 5000 is in use
ss -tlnp | grep 5000

# Or
lsof -i :5000
```

**Solution:**
```bash
# Restart the bot (dashboard restarts with it)
julaba restart

# Or force restart
pkill -9 -f "python3 bot.py"
rm -f /home/opc/julaba/julaba.lock
julaba start
```

---

### Problem: Bot Crashes on Startup

**Check logs for errors:**
```bash
# View recent log entries
tail -100 /home/opc/julaba/bot.log

# Or if using systemd
sudo journalctl -u julaba -n 100

# Look for specific errors
grep -i "error\|exception\|traceback" /home/opc/julaba/bot.log | tail -20
```

**Common fixes:**

1. **Syntax error in config:**
```bash
# Validate JSON config
python3 -c "import json; json.load(open('/home/opc/julaba/julaba_config.json'))"
```

2. **Missing dependencies:**
```bash
cd /home/opc/julaba
pip3 install -r requirements.txt
```

3. **Exchange API issues:**
```bash
# Check if API keys are set
grep -E "api_key|secret" /home/opc/julaba/julaba_config.json
```

---

### Problem: Positions Not Showing

**Symptoms:**
- Dashboard shows no positions but trade was opened
- Position data missing after restart

**Check position state file:**
```bash
# View saved trading state
cat /home/opc/julaba/trading_state.json | python3 -m json.tool
```

**Manual position check via API:**
```bash
curl -s http://localhost:5000/api/data | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Open Position:', d.get('open_position'))
print('Additional:', d.get('additional_positions'))
"
```

---

### Problem: High Memory Usage

**Check memory usage:**
```bash
# Memory used by bot process
ps aux | grep "python3 bot.py" | awk '{print "Memory: " $6/1024 " MB"}'

# System memory
free -h
```

**Solution - Restart to clear memory:**
```bash
julaba restart
```

---

### Problem: Exchange Connection Failed

**Symptoms:**
- "Exchange not connected" errors
- Order execution failures

**Check connectivity:**
```bash
# Test Bybit API
curl -s "https://api.bybit.com/v5/market/time"
# Should return server time

# Check your IP
curl -s ifconfig.me
```

**Solution:**
- Verify API keys in config
- Check if IP is whitelisted on exchange
- Restart bot to reconnect

---

## 🛠️ Advanced Commands

### Manual Position Close via API

```bash
# Close specific position
curl -X POST -H "Content-Type: application/json" \
  -d '{"symbol": "ETHUSDT"}' \
  http://localhost:5000/api/control/close-position

# Close all positions (one by one)
curl -X POST -H "Content-Type: application/json" \
  -d '{"symbol": "ETHUSDT"}' \
  http://localhost:5000/api/control/close-position

curl -X POST -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT"}' \
  http://localhost:5000/api/control/close-position
```

### Pause/Resume Trading

```bash
# Pause trading (keeps position management)
curl -X POST http://localhost:5000/api/control/pause

# Resume trading
curl -X POST http://localhost:5000/api/control/resume
```

### Get Full System State

```bash
# Complete state dump
curl -s http://localhost:5000/api/data | python3 -m json.tool

# Just positions
curl -s http://localhost:5000/api/data | python3 -c "
import sys, json
d = json.load(sys.stdin)
import pprint
pprint.pprint({
    'positions': [d.get('open_position')] + d.get('additional_positions', []),
    'balance': d.get('balance'),
    'status': d.get('status')
})
"
```

### Process Management

```bash
# Find bot PID
pgrep -f "python3 bot.py"

# Watch bot resource usage in real-time
watch -n 1 'ps aux | grep "python3 bot.py" | grep -v grep'

# Monitor CPU/Memory
top -p $(pgrep -f "python3 bot.py" | head -1)
```

---

## 📁 Important File Locations

| File | Purpose |
|------|---------|
| `/home/opc/julaba/bot.py` | Main bot code |
| `/home/opc/julaba/julaba_config.json` | Configuration |
| `/home/opc/julaba/julaba.lock` | Lock file (prevents multiple instances) |
| `/home/opc/julaba/trading_state.json` | Saved position state |
| `/home/opc/julaba/trade_history.json` | Trade history |
| `/home/opc/julaba/bot.log` | Log file |
| `/home/opc/julaba/ai_decisions.json` | AI decision history |
| `/home/opc/julaba/adaptive_params.json` | Adaptive parameters |

---

## 🚨 Emergency Commands

### EMERGENCY: Stop Everything NOW
```bash
pkill -9 -f "python3 bot.py" && rm -f /home/opc/julaba/julaba.lock
echo "Bot killed and lock removed"
```

### EMERGENCY: Close All Positions & Stop
```bash
# Close position 1
curl -X POST -H "Content-Type: application/json" \
  -d '{"symbol": "ETHUSDT"}' \
  http://localhost:5000/api/control/close-position

# Close position 2
curl -X POST -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT"}' \
  http://localhost:5000/api/control/close-position

# Then stop bot
julaba stop
```

### EMERGENCY: Full Reset
```bash
# Kill everything
pkill -9 -f "python3 bot.py"

# Remove all lock and state files
rm -f /home/opc/julaba/julaba.lock
rm -f /home/opc/julaba/trading_state.json

# Fresh start
julaba start
```

---

## 🔄 Systemd Service Commands (if using service)

```bash
# Start service
sudo systemctl start julaba

# Stop service
sudo systemctl stop julaba

# Restart service
sudo systemctl restart julaba

# Check service status
sudo systemctl status julaba

# View service logs
sudo journalctl -u julaba -f

# Enable service on boot
sudo systemctl enable julaba

# Disable service on boot
sudo systemctl disable julaba
```

---

## 📊 Quick Health Check Script

Run this to check everything:

```bash
echo "=== JULABA HEALTH CHECK ==="
echo ""

# Check process
if pgrep -f "python3 bot.py" > /dev/null; then
    echo "✓ Bot process: RUNNING (PID: $(pgrep -f 'python3 bot.py' | head -1))"
else
    echo "✗ Bot process: NOT RUNNING"
fi

# Check lock file
if [ -f /home/opc/julaba/julaba.lock ]; then
    echo "✓ Lock file: EXISTS"
else
    echo "! Lock file: MISSING"
fi

# Check API
if curl -s --max-time 3 http://localhost:5000/api/data | grep -q "balance"; then
    echo "✓ Dashboard API: RESPONDING"
else
    echo "✗ Dashboard API: NOT RESPONDING"
fi

# Check disk
DISK_PCT=$(df -h /home/opc | tail -1 | awk '{print $5}')
echo "ℹ Disk usage: $DISK_PCT"

# Check memory
MEM_PCT=$(free | grep Mem | awk '{printf "%.0f%%", $3/$2 * 100}')
echo "ℹ Memory usage: $MEM_PCT"

# Show positions
echo ""
echo "=== POSITIONS ==="
curl -s http://localhost:5000/api/data 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    op = d.get('open_position')
    add = d.get('additional_positions', [])
    if op:
        print(f'  1. {op.get(\"symbol\")} {op.get(\"side\")} | PnL: \${op.get(\"pnl\",0):.2f}')
    for i, p in enumerate(add):
        print(f'  {i+2}. {p.get(\"symbol\")} {p.get(\"side\")} | PnL: \${p.get(\"pnl\",0):.2f}')
    if not op and not add:
        print('  No open positions')
except:
    print('  Unable to fetch')
" 2>/dev/null || echo "  Unable to fetch"
```

---

## 💡 Tips

1. **Always check status first:**
   ```bash
   julaba status
   ```

2. **Before restarting, check if positions are open:**
   ```bash
   julaba positions
   ```

3. **If something is wrong, check logs:**
   ```bash
   julaba logs error
   ```

4. **Regular health checks:**
   ```bash
   julaba health
   ```

5. **Clean restart if issues persist:**
   ```bash
   pkill -9 -f "python3 bot.py"
   rm -f /home/opc/julaba/julaba.lock
   sleep 2
   julaba start
   julaba status
   ```

---

*Last updated: January 2026*
*Julaba Trading Bot v2.0*
