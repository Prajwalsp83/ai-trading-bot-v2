#!/bin/bash
# Quick status check for the running paper bot.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$PROJECT_DIR/data/paper.log"
TRADES_CSV="$PROJECT_DIR/data/paper_trades.csv"
STATE_FILE="$PROJECT_DIR/.paper_state.json"
LABEL="com.psp.ai-trading-bot.paper"

echo ""
echo "================================================"
echo " AI Trading Bot - Status"
echo "================================================"

# launchd
echo ""
echo "1. Launchd registration:"
if launchctl list | grep -q "ai-trading-bot"; then
    launchctl list | grep "ai-trading-bot" | sed 's/^/   /'
    echo "   (PID column != '-' means running; '0' in 2nd col means last exit was clean)"
else
    echo "   NOT RUNNING (not registered with launchd)"
fi

# state
echo ""
echo "2. State:"
if [ -f "$STATE_FILE" ]; then
    python3 -c "
import json
s = json.load(open('$STATE_FILE'))
print(f\"   equity     : {s.get('equity', 0):,.2f}\")
print(f\"   peak equity: {s.get('peak_equity', 0):,.2f}\")
op = s.get('open_position')
if op:
    print(f\"   open       : {op['side']} @ {op['entry']:.2f}  sl={op['sl']:.2f}  tp={op['tp']:.2f}\")
else:
    print(f\"   open       : flat\")
print(f\"   today P&L  : {s.get('pnl_today', 0):+.2f} over {s.get('trades_today', 0)} trades\")
print(f\"   last bar   : {s.get('last_bar_ts', '(none yet)')}\")
"
else
    echo "   (state file not created yet)"
fi

# trades
echo ""
echo "3. Trade journal:"
if [ -f "$TRADES_CSV" ]; then
    LINES=$(( $(wc -l < "$TRADES_CSV") - 1 ))
    echo "   closed trades: $LINES"
    if [ "$LINES" -gt 0 ]; then
        echo "   last 3:"
        tail -n 3 "$TRADES_CSV" | awk -F',' '{ printf "     #%s  %s  %s -> %s  pnl=%s  R=%s  reason=%s\n", $1, $4, $5, $6, $10, $11, $14 }'
    fi
else
    echo "   (no trades yet)"
fi

# log tail
echo ""
echo "4. Last 8 log lines:"
if [ -f "$LOG_FILE" ]; then
    tail -n 8 "$LOG_FILE" | sed 's/^/   /'
else
    echo "   (no log yet)"
fi

echo ""
echo "================================================"
