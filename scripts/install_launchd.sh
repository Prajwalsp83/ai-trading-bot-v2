#!/bin/bash
# Install AI Trading Bot as a macOS launchd user agent.
# Bot auto-starts at login, auto-restarts on crash, logs to data/paper.log.
#
# Run from anywhere:
#     bash ~/Documents/ai-trading-bot/v2/scripts/install_launchd.sh
#
# Re-running is safe — it unloads any previous version first.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
SCRIPT="$PROJECT_DIR/scripts/paper_live.py"
LOG_FILE="$PROJECT_DIR/data/paper.log"
LABEL="com.psp.ai-trading-bot.paper"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"

echo ""
echo "==============================================="
echo " AI Trading Bot - launchd installer"
echo "==============================================="
echo " project : $PROJECT_DIR"
echo " python  : $PYTHON_BIN"
echo " script  : $SCRIPT"
echo " log     : $LOG_FILE"
echo " label   : $LABEL"
echo "==============================================="
echo ""

# ---- pre-flight ----
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: venv python not found at $PYTHON_BIN"
    echo "Did you create the venv? cd $PROJECT_DIR && python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: paper_live.py not found at $SCRIPT"
    exit 1
fi
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "ERROR: .env file not found at $PROJECT_DIR/.env"
    echo "Create it first (see Step 3 in our chat) — bot needs Telegram + Upstox keys."
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$HOME/Library/LaunchAgents"

# ---- unload existing version if any ----
if [ -f "$PLIST_PATH" ]; then
    echo "Found existing plist - unloading first..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
fi

# ---- generate plist ----
cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>$SCRIPT</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>

    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin</string>
        <key>HOME</key>
        <string>$HOME</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
EOF

chmod 644 "$PLIST_PATH"
echo "Wrote plist -> $PLIST_PATH"

# ---- load ----
launchctl load "$PLIST_PATH"
echo "Loaded into launchd"

# ---- verify ----
sleep 2
echo ""
echo "Status:"
if launchctl list | grep -q "ai-trading-bot"; then
    launchctl list | grep "ai-trading-bot"
    echo ""
    echo "Last 5 log lines:"
    sleep 1
    if [ -f "$LOG_FILE" ]; then
        tail -n 5 "$LOG_FILE" | sed 's/^/  /'
    else
        echo "  (log file not created yet - check again in 30s)"
    fi
else
    echo "  WARNING: not found in launchctl list. Check log: $LOG_FILE"
fi

echo ""
echo "=============================================="
echo " Done. The bot is now running in background."
echo ""
echo " To watch live logs:"
echo "   tail -f $LOG_FILE"
echo ""
echo " To check status:"
echo "   bash $PROJECT_DIR/scripts/status.sh"
echo ""
echo " To stop / uninstall:"
echo "   bash $PROJECT_DIR/scripts/uninstall_launchd.sh"
echo "=============================================="
