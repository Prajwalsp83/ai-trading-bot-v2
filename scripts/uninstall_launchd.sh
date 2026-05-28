#!/bin/bash
# Stop and uninstall the AI Trading Bot launchd agent.
# Trade journal and state file are preserved.

set -e

LABEL="com.psp.ai-trading-bot.paper"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ ! -f "$PLIST_PATH" ]; then
    echo "Not installed (no plist at $PLIST_PATH)"
    exit 0
fi

launchctl unload "$PLIST_PATH" 2>/dev/null || true
rm -f "$PLIST_PATH"

echo "Unloaded and removed $LABEL."
echo "Your trade journal (data/paper_trades.csv) and state (.paper_state.json) are preserved."
