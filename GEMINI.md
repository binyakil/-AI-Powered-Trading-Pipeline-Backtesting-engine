# Project
Algorithmic trading system written in Python.

# Architecture Rules
- Keep the backtesting engine (`backtesting_engine/`) strictly separated from the live bot deployments (`deployments/julaba_bot/`).
- The indicator math must be identical across backtesting and live environments.
- Do not modify core logic without explaining the mathematical reason first.