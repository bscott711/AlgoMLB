# Module Specification: Future Layer Stubs

The following modules represent the roadmap for the AlgoMLB project. They are currently implemented as stubs (`__init__.py` only) to reserve namespace and define architectural boundaries.

## 1. Strategy Layer (`algomlb.strategy`)

**Current State**: Stub
**Objective**: Transform ML model outputs into actionable betting decisions.

### Proposed Responsibilities:
- **Bankroll Management**: Implementation of Kelly Criterion and fractional betting logic.
- **Probability Arbitrage**: Identifying discrepancies between model fair-value and market prices.
- **Backtesting Engine**: Simulating historical PnL based on model checkpoints and historical closing odds.

---

## 2. Execution Layer (`algomlb.execution`)

**Current State**: Stub
**Objective**: Automate the interaction with market liquidity.

### Proposed Responsibilities:
- **Bookmaker Integration**: Wrapper for betting exchange APIs (e.g., Circa, FanDuel).
- **Order Management**: Tracking pending, confirmed, and settled bets in a persistent ledger.
- **Safety Valves**: Kill-switch logic to halt execution if model performance deviates from expected variance.

---

## 3. Social Layer (`algomlb.social`)

**Current State**: Stub
**Objective**: Automated reporting and community distribution.

### Proposed Responsibilities:
- **Discord/Slack Bots**: Real-time alerts for "Model Value" detections.
- **Progress Reports**: Daily PnL and system health updates via automated social posts.
- **Feeds**: Standardized JSON/RSS exports for downstream consumers.

## Engineering Notes
- **Philosophy**: These modules should remain isolated from the `domain` and `ml` core to allow for modular experimentation without risking data integrity.
- **Next Epic**: Once the `ml` layer is fully calibrated, the `strategy` module will be the first stub to be fleshed out.
