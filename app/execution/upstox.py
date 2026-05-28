"""
Upstox v2 API broker (MCX gold futures: GOLDM).

Auth: OAuth 2.0 authorization code flow.
  1. User opens login URL (we open it in default browser)
  2. After login, Upstox redirects to http://127.0.0.1:5000/callback?code=XYZ
  3. Local Flask listener captures the code
  4. Bot exchanges code for access_token (valid until ~3:30 AM IST next day)
  5. Access token is cached locally; refreshed daily via re-login

Daily re-login is a SEBI/Upstox compliance requirement (not a v1 limitation).
We surface a Telegram alert 30 min before token expiry so user can approve.

Skeleton — no implementation yet.
"""
from typing import Literal

from ..core.events import Fill, Order, Position
from .base import Broker

UpstoxEnv = Literal["sandbox", "live"]


class UpstoxBroker(Broker):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        redirect_uri: str = "http://127.0.0.1:5000/callback",
        environment: UpstoxEnv = "live",
        max_account_equity_inr: float | None = None,
    ) -> None:
        """
        max_account_equity_inr: HARD CAP enforced before every order.
        If account equity > cap, refuse new orders. Small-account guard.
        Default placeholder in config = ₹50,000.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.redirect_uri = redirect_uri
        self.environment = environment
        self.max_account_equity_inr = max_account_equity_inr
        self._access_token: str | None = None
        self._token_expires_at = None
        # TODO: lazy import upstox_client (official SDK).

    # ---- auth ----
    def login_url(self) -> str:
        """TODO: build https://api.upstox.com/v2/login/authorization/dialog URL
        with client_id=self.api_key, redirect_uri, response_type=code, state."""
        raise NotImplementedError

    def exchange_code(self, code: str) -> None:
        """TODO: POST /v2/login/authorization/token; cache access_token + expiry."""
        raise NotImplementedError

    def ensure_authenticated(self) -> None:
        """TODO: if no token or expired -> raise + publish RISK_ALERT for user re-login."""
        raise NotImplementedError

    # ---- broker interface ----
    def place(self, order: Order) -> str:
        """TODO:
          1. _enforce_equity_cap()
          2. POST /v2/order/place with:
             - instrument_token: NSE/MCX:GOLDM<YYMM>FUT
             - quantity, price (0 for market), order_type=MARKET
             - product: I (intraday) or D (delivery)
             - is_amo: false
          3. After fill confirm, place a separate SL order (bracket-equivalent —
             Upstox doesn't support OCO directly on MCX, so we manage SL/TP
             as separate orders in OrderManager)
        """
        raise NotImplementedError

    def modify(self, order_id: str, sl: float | None = None, tp: float | None = None) -> None:
        """TODO: PUT /v2/order/modify for SL order; cancel + replace TP."""
        raise NotImplementedError

    def cancel(self, order_id: str) -> None:
        """TODO: DELETE /v2/order/cancel."""
        raise NotImplementedError

    def positions(self) -> list[Position]:
        """TODO: GET /v2/portfolio/short-term-positions; map to Position list."""
        raise NotImplementedError

    def account_equity(self) -> float:
        """TODO: GET /v2/user/get-funds-and-margin; sum across segments."""
        raise NotImplementedError

    def on_fill(self, callback) -> None:
        """TODO: subscribe to Upstox websocket order updates feed; dispatch fills."""
        raise NotImplementedError

    def _enforce_equity_cap(self) -> None:
        """TODO: if account_equity() > max_account_equity_inr: trip kill switch
        with reason 'equity cap reached: scale-up requires re-validation'."""
        raise NotImplementedError
