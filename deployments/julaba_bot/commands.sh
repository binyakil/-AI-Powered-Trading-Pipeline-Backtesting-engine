#!/bin/bash
# Julaba Trading Bot - Simple Commands
# Usage: j [command]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat << EOF
JULABA BOT COMMANDS (usage: j <cmd>)

SYSTEM:
  s, status       Service status
  start           Start bot
  stop            Stop bot  
  r, restart      Restart bot
  l, logs         Live logs
  la              AI logs
  lt              Trade logs

QUICK INFO:
  b               Balance
  p               Position
  ai              AI mode
  sym             Symbol
  pnl             P&L

ANALYSIS:
  perf            Performance
  ml              ML stats
  trades          Last trades
  err             Recent errors

TOOLS:
  check           Health check
  backup          Run backup
  disk            Disk space
  tail            Tail log file

Examples:
  j s             # status
  j b             # balance
  j la            # AI logs
  j perf          # performance
  
EOF
}

cmd_status() {
    sudo systemctl status julaba --no-pager
}

cmd_start() {
    sudo systemctl start julaba
    sleep 1
    sudo systemctl status julaba --no-pager
}

cmd_stop() {
    sudo systemctl stop julaba
}

cmd_restart() {
    sudo systemctl restart julaba
    sleep 1
    sudo systemctl status julaba --no-pager
}

cmd_logs() {
    sudo journalctl -u julaba -f
}

cmd_logs_ai() {
    sudo journalctl -u julaba -f | grep -E "AI|AUTONOMOUS|🤖|🧠"
}

cmd_logs_trades() {
    sudo journalctl -u julaba -f | grep -E "LONG|SHORT|Position|Entry|Exit"
}

# Monitoring
cmd_check() {
    [ -f "$SCRIPT_DIR/health_check.sh" ] && bash "$SCRIPT_DIR/health_check.sh" || echo "health_check.sh not found"
}

cmd_ai_mode() {
    curl -s http://localhost:5000/api/data | jq -r '.params.ai_mode // "N/A"'
}

cmd_position() {
    curl -s http://localhost:5000/api/data | jq '.position // "No position"'
}

cmd_balance() {
    curl -s http://localhost:5000/api/data | jq '.balance'
}

cmd_pnl() {
    curl -s http://localhost:5000/api/data | jq '.pnl'
}

cmd_symbol() {
    curl -s http://localhost:5000/api/data | jq -r '.status.symbol // "N/A"'
}

# Data & Analysis
cmd_ml_stats() {
    curl -s http://localhost:5000/api/data | jq '.ml'
}

cmd_performance() {
    curl -s http://localhost:5000/api/data | jq '{balance, pnl, ai: .ai.stats, ml}'
}

cmd_last_trades() {
    [ -f "$SCRIPT_DIR/trade_history.json" ] && jq -r '.trades[-10:] | .[] | "\(.timestamp) | \(.side) \(.symbol) | PNL: \(.pnl)"' "$SCRIPT_DIR/trade_history.json" || echo "No trades"
}

# Maintenance
cmd_backup() {
    [ -f "$SCRIPT_DIR/backup.sh" ] && bash "$SCRIPT_DIR/backup.sh" || echo "backup.sh not found"
}

cmd_disk() {
    df -h /home/opc | tail -1
    echo ""
    du -sh "$SCRIPT_DIR"
}

cmd_errors() {
    sudo journalctl -u julaba -n 50 | grep -i "error\|exception\|critical" || echo "No errors"
}

cmd_tail() {
    tail -f "$SCRIPT_DIR/julaba.log"
}

# Main command dispatcher
case "$1" in
    s|status) cmd_status ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    r|restart) cmd_restart ;;
    l|logs) cmd_logs ;;
    la) cmd_logs_ai ;;
    lt) cmd_logs_trades ;;
    
    b) cmd_balance ;;
    p) cmd_position ;;
    ai) cmd_ai_mode ;;
    sym) cmd_symbol ;;
    pnl) cmd_pnl ;;
    
    perf) cmd_performance ;;
    ml) cmd_ml_stats ;;
    trades) cmd_last_trades ;;
    err) cmd_errors ;;
    
    check) cmd_check ;;
    backup) cmd_backup ;;
    disk) cmd_disk ;;
    tail) cmd_tail ;;
    
    *) show_help ;;
esac
