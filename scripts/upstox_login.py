"""
Upstox OAuth login helper.

Run once per day (Upstox tokens expire at ~03:30 IST regardless of when issued).

What it does:
  1. Loads UPSTOX_API_KEY + UPSTOX_API_SECRET from .env
  2. Starts a tiny local Flask server on http://127.0.0.1:5000
  3. Opens the Upstox login page in your default browser
  4. Captures the OAuth `code` Upstox redirects back with
  5. Exchanges the code for an access_token
  6. Saves the token to .upstox_token.json (gitignored)
  7. Exits

Run:
    cd ~/Documents/ai-trading-bot/v2
    source .venv/bin/activate
    python scripts/upstox_login.py
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from flask import Flask, request


# ---- config ----
HERE = Path(__file__).resolve().parent.parent          # v2/
TOKEN_FILE = HERE / ".upstox_token.json"
REDIRECT_URI = "http://127.0.0.1:5000/callback"
AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"

# ---- shared state between Flask thread and main thread ----
result: dict = {"code": None, "error": None, "state_sent": None}


def make_app() -> Flask:
    app = Flask(__name__)

    @app.get("/callback")
    def callback():
        # CSRF check
        state = request.args.get("state")
        if state != result["state_sent"]:
            result["error"] = f"state mismatch: got {state!r}"
            return _html("Login failed", "State mismatch — possible CSRF. Try again.", ok=False), 400

        # Upstox returns either code (success) or error
        code = request.args.get("code")
        err = request.args.get("error")
        if err:
            result["error"] = err
            return _html("Login failed", f"Upstox returned: {err}", ok=False), 400
        if not code:
            result["error"] = "no code in callback"
            return _html("Login failed", "No code in callback URL.", ok=False), 400

        result["code"] = code
        return _html("Login successful", "You can close this tab. Back to the terminal.", ok=True)

    @app.get("/")
    def index():
        return "OAuth listener running. Waiting for Upstox callback…"

    return app


def _html(title: str, body: str, ok: bool) -> str:
    color = "#16a34a" if ok else "#dc2626"
    return f"""
    <html><head><title>{title}</title>
    <style>
      body {{ font-family: -apple-system, system-ui, sans-serif;
              max-width: 480px; margin: 80px auto; padding: 0 20px;
              color: #111; }}
      .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px;
                background: {color}; color: white; font-size: 12px;
                letter-spacing: 0.05em; }}
      h1 {{ margin: 12px 0; }}
    </style></head>
    <body>
      <span class="badge">{title.upper()}</span>
      <h1>{title}</h1>
      <p>{body}</p>
    </body></html>
    """


def run_server_in_thread(app: Flask) -> threading.Thread:
    """Start Flask in a daemon thread so main can wait for the code."""
    def _serve():
        # quiet werkzeug
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


def main() -> int:
    load_dotenv(HERE / ".env")
    api_key = os.getenv("UPSTOX_API_KEY")
    api_secret = os.getenv("UPSTOX_API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: UPSTOX_API_KEY or UPSTOX_API_SECRET missing from .env", file=sys.stderr)
        return 1

    # 1. CSRF state
    state = secrets.token_urlsafe(16)
    result["state_sent"] = state

    # 2. Build login URL
    params = {
        "client_id": api_key,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "state": state,
    }
    login_url = f"{AUTH_URL}?{urlencode(params)}"

    # 3. Start local listener
    app = make_app()
    run_server_in_thread(app)
    time.sleep(0.3)  # let Flask bind

    # 4. Open browser
    print("\nOpening Upstox login in your browser…")
    print(f"If it doesn't open, paste this into your browser:\n  {login_url}\n")
    webbrowser.open(login_url)

    # 5. Wait for callback
    print("Waiting for callback on http://127.0.0.1:5000/callback …")
    print("(Log in to Upstox in the browser tab that just opened.)\n")
    deadline = time.time() + 300  # 5 min
    while result["code"] is None and result["error"] is None and time.time() < deadline:
        time.sleep(0.5)

    if result["error"]:
        print(f"\nLogin failed: {result['error']}", file=sys.stderr)
        return 2
    if result["code"] is None:
        print("\nLogin timed out after 5 minutes.", file=sys.stderr)
        return 3

    code = result["code"]
    print("Got authorization code. Exchanging for access token…")

    # 6. Exchange code for token
    r = requests.post(
        TOKEN_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "code": code,
            "client_id": api_key,
            "client_secret": api_secret,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=20,
    )

    if r.status_code != 200:
        print(f"\nToken exchange failed: HTTP {r.status_code}\n{r.text}", file=sys.stderr)
        return 4

    payload = r.json()
    if "access_token" not in payload:
        print(f"\nNo access_token in response:\n{json.dumps(payload, indent=2)}", file=sys.stderr)
        return 5

    # 7. Save token
    token_record = {
        "access_token": payload["access_token"],
        "user_id": payload.get("user_id"),
        "user_name": payload.get("user_name"),
        "email": payload.get("email"),
        "broker": payload.get("broker"),
        "exchanges": payload.get("exchanges", []),
        "products": payload.get("products", []),
        "issued_at": datetime.now().isoformat(timespec="seconds"),
    }
    TOKEN_FILE.write_text(json.dumps(token_record, indent=2))
    TOKEN_FILE.chmod(0o600)

    print(f"\n✓ Token saved to {TOKEN_FILE.relative_to(HERE.parent)}")
    print(f"  user_id  : {token_record['user_id']}")
    print(f"  exchanges: {', '.join(token_record['exchanges'])}")
    print(f"  expires  : ~03:30 IST tomorrow (Upstox policy)\n")
    print("Next: python scripts/upstox_smoke.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
