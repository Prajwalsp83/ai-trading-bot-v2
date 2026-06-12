"""
One-shot, zero-babysitting validation suite. Run it in the background on the
VPS, close RDP, and get the verdict on Telegram when it finishes.

Runs, in order (continuing past failures so one bad step doesn't kill the rest):
  1. fvg_scalp backtest      (the new scalper's first real numbers)
  2. walk_forward_oos.py     (SMC out-of-sample verdict -- the heavy step)
  3. gen_review_report.py    (before/after markdown report)

Skipped on purpose: mean_reversion + liquidity_sweep re-runs (both documented
negative-edge and disabled live -- nothing rides on re-validating them) and the
standalone sweep (walk_forward_oos runs the sweep per train window itself).

Launch on the VPS so it survives closing the RDP *window* (do NOT sign out):
    cd C:\\ai-trading-bot
    Start-Process -FilePath python -ArgumentList "scripts\\run_validation_suite.py" `
        -RedirectStandardOutput "logs\\validation.out.log" `
        -RedirectStandardError  "logs\\validation.err.log" -WindowStyle Hidden

Watch progress any time:
    Get-Content .\\logs\\validation.out.log -Tail 20
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SCRIPTS = HERE / "scripts"
BT_DIR = HERE / "data" / "backtests"

STEPS = [
    ("fvg_scalp backtest",
     ["run_backtest.py", "--strategy", "fvg_scalp", "--years", "4",
      "--equity", "960", "--risk-pct", "0.02"]),
    ("SMC OOS walk-forward",
     ["walk_forward_oos.py", "--train-months", "12", "--test-months", "6"]),
    ("review report",
     ["gen_review_report.py"]),
]


def _env_value(key: str) -> str | None:
    """Minimal .env reader -- no python-dotenv dependency needed."""
    env_path = HERE / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def tg_send(text: str) -> None:
    """Best-effort Telegram notify; never raises."""
    token = _env_value("TELEGRAM_BOT_TOKEN")
    chat_id = _env_value("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[notify] telegram creds missing in .env; skipping notify")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    # Prefer requests: it ships its own CA bundle (certifi), which is what the
    # live bots use successfully on the VPS. Plain urllib hit
    # CERTIFICATE_VERIFY_FAILED there (system trust store issue).
    try:
        import requests
        requests.post(url, data=payload, timeout=15).raise_for_status()
        print("[notify] telegram message sent (requests)")
        return
    except Exception as e:
        print(f"[notify] requests path failed ({e}); trying urllib")
    try:
        data = urllib.parse.urlencode(payload).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15)
        print("[notify] telegram message sent (urllib)")
    except Exception as e:
        print(f"[notify] telegram send failed (non-fatal): {e}")


def _latest_summary(strategy: str) -> dict | None:
    paths = sorted(glob.glob(str(BT_DIR / f"{strategy}_*_summary.json")))
    if not paths:
        return None
    try:
        return json.load(open(paths[-1]))
    except Exception:
        return None


def _oos_report() -> dict | None:
    p = BT_DIR / "walk_forward_oos_report.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _fmt_fvg(s: dict | None) -> str:
    if not s:
        return "fvg_scalp: NO RESULT (step failed?)"
    m = s.get("metrics", s)  # summary json may nest metrics or be flat
    trades = m.get("trades", 0)
    if not trades:
        return "fvg_scalp: 0 trades"
    per_day = trades / (4.0 * 252.0)
    return (f"fvg_scalp 4yr: {trades} trades (~{per_day:.1f}/day), "
            f"WR {m.get('win_rate_pct', 0):.1f}%, PF {m.get('profit_factor')}, "
            f"PnL {m.get('net_pnl_pct', 0):+.1f}%, maxDD {m.get('max_dd_pct', 0):.1f}%")


def _fmt_oos(r: dict | None) -> str:
    if not r:
        return "SMC OOS: NO RESULT (step failed?)"
    o = r.get("out_of_sample_overall", {})
    pf = o.get("profit_factor")
    try:
        deployable = float(pf) >= 1.2
    except (TypeError, ValueError):
        deployable = pf == "inf"
    verdict = "DEPLOYABLE (OOS PF >= 1.2)" if deployable else "NOT DEPLOYABLE (OOS PF < 1.2)"
    return (f"SMC OOS: PF {pf}, total R {o.get('total_r', 0):+.1f}, "
            f"{r.get('positive_oos_windows')}/{r.get('n_windows')} windows positive, "
            f"WR {o.get('win_rate', 0)*100:.1f}%, maxDD {o.get('max_dd_r', 0):.1f}R\n"
            f"Verdict: {verdict}")


def main() -> int:
    t0 = time.time()
    print("=== VALIDATION SUITE START ===", flush=True)
    failures = []
    for name, argv in STEPS:
        print(f"\n--- step: {name} ---", flush=True)
        t = time.time()
        try:
            # PYTHONIOENCODING=utf-8: when stdout is redirected to a file on
            # Windows, Python defaults to cp1252 and any non-ASCII char in a
            # child's print crashes it (this killed the fvg_scalp step once).
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            rc = subprocess.run(
                [sys.executable, str(SCRIPTS / argv[0])] + argv[1:],
                cwd=HERE, env=env).returncode
        except Exception as e:
            print(f"step crashed: {e}", flush=True)
            rc = -1
        mins = (time.time() - t) / 60
        print(f"--- step done: {name} rc={rc} ({mins:.1f} min) ---", flush=True)
        if rc != 0:
            failures.append(name)

    hours = (time.time() - t0) / 3600
    lines = [f"VALIDATION SUITE {'DONE' if not failures else 'DONE WITH FAILURES'} "
             f"({hours:.1f}h)"]
    if failures:
        lines.append("Failed steps: " + ", ".join(failures)
                      + " -- check logs\\validation.out.log")
    lines.append(_fmt_fvg(_latest_summary("fvg_scalp")))
    lines.append(_fmt_oos(_oos_report()))
    lines.append("Full report: data/backtests/review_fixes_report.md")
    msg = "\n".join(lines)
    print("\n" + msg, flush=True)
    tg_send(msg)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
