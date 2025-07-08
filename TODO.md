# Optimization To-Do List

This document tracks the progress of optimizing various portfolio backtester scenarios.

## Scenarios to Optimize:

- [ ] **Momentum_Unfiltered**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Momentum_Unfiltered`
  - Output Analysis: (To be filled after run)

- [ ] **Momentum_Unfiltered_ATR**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Momentum_Unfiltered_ATR`
  - Output Analysis: (To be filled after run)

- [ ] **FilteredLaggedMomentum_Optimization**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name FilteredLaggedMomentum_Optimization`
  - Output Analysis: (To be filled after run)

- [ ] **FilteredLaggedMomentum_Test**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name FilteredLaggedMomentum_Test`
  - Output Analysis: (To be filled after run)

- [ ] **Momentum_Beta_Sized**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Momentum_Beta_Sized`
  - Output Analysis: (To be filled after run)

- [ ] **Momentum_DVOL_Sizer**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Momentum_DVOL_Sizer`
  - Output Analysis: (To be filled after run)

- [ ] **Momentum_SMA_Filtered**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Momentum_SMA_Filtered`
  - Output Analysis: (To be filled after run)

- [ ] **Sharpe_Momentum**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Sharpe_Momentum`
  - Output Analysis: (To be filled after run)

- [ ] **Test_Optuna_Minimal**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Test_Optuna_Minimal`
  - Output Analysis: (To be filled after run)

- [ ] **Test_Genetic_Minimal**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Test_Genetic_Minimal`
  - Output Analysis: (To be filled after run)

- [ ] **Momentum_Optimize_Sizer**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Momentum_Optimize_Sizer`
  - Output Analysis: (To be filled after run)

- [ ] **VAMS_Downside_Penalized**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name VAMS_Downside_Penalized`
  - Output Analysis: (To be filled after run)

- [ ] **VAMS_No_Downside**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name VAMS_No_Downside`
  - Output Analysis: (To be filled after run)

- [ ] **Sortino_Momentum**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Sortino_Momentum`
  - Output Analysis: (To be filled after run)

- [ ] **Calmar_Momentum**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Calmar_Momentum`
  - Output Analysis: (To be filled after run)

- [ ] **Sharpe_Sized_Momentum**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Sharpe_Sized_Momentum`
  - Output Analysis: (To be filled after run)

- [ ] **FilteredLaggedMomentum_Simple**
  - Status: Pending
  - Command: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-name FilteredLaggedMomentum_Simple`
  - Output Analysis: (To be filled after run)

## Instructions:

1.  **Run Optimization:** Copy and paste the `Command` for the next pending scenario into your terminal and execute it.
2.  **Analyze Output:** Carefully review the terminal output for any errors, warnings, signs of missing data, infinity/zero/NaN reads, or extremely optimistic/pessimistic results.
3.  **Update TODO.md:** After each run, update the `TODO.md` file by:
    *   Changing `[ ]` to `[x]` for the completed scenario.
    *   Adding a summary of the `Output Analysis` (e.g., "No issues found," "Warning: X occurred," "Error: Y encountered").
4.  **Address Issues:** If problems are found, diagnose and fix them before proceeding to the next scenario.