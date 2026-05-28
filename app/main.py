"""
Orchestrator. The live event loop.

Wiring (skeleton):

  config       = load_config("config.yaml")
  bus          = EventBus()
  state        = StateStore()
  kill         = KillSwitch(flag_file=...)
  data         = YFinanceProvider(...) | MT5Provider(...)
  strategy     = BreakoutTrendStrategy(...)
  risk_gate    = RiskGate(state, config.risk)
  broker       = PaperBroker(...) | MT5Broker(...)
  order_mgr    = OrderManager(broker, kill)
  journal      = TradeJournal(config.journal_path)
  router       = AlertRouter({"telegram": ..., "sound": ..., "desktop": ...})

  bus.subscribe(Alert,  router.handle)
  bus.subscribe(Fill,   journal_recorder)
  bus.subscribe(Fill,   state_updater)

  data.subscribe_live(symbol, on_new_bar)

  on_new_bar(bars):
      market_state = MarketState(...)
      signal = strategy.evaluate(market_state)
      if signal:
          bus.publish(Alert.from_signal(signal))
          if signal.severity in (BUY_READY, SELL_READY):
              gate = risk_gate.check_all()
              if not gate.allowed:
                  bus.publish(Alert(RISK_ALERT, gate.reason))
                  return
              sl, tp = initial_sl_tp(...)
              qty    = position_size(...)
              order  = Order(symbol, side, qty, sl, tp)
              order_mgr.submit(order)

      for pos in broker.positions():
          new_sl = update_trailing(pos, last_price, atr, stops_cfg)
          if new_sl is not None:
              order_mgr.update_stops(pos, new_sl, None)

Skeleton — no implementation yet.
"""


def main() -> None:
    """TODO: build the wiring above and start the data subscription."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
