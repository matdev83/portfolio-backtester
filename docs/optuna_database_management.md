# Optuna Database Management

## Overview

The Portfolio Backtester uses Optuna for Bayesian optimization, with all study data stored in a SQLite database located at `data/optuna_studies.db`. This document explains how to manage and work with Optuna studies.

## Database Location

- **File Path:** `data/optuna_studies.db`
- **Format:** SQLite database
- **Auto-Creation:** The database is automatically created when the first optimization study is run

## Study Management

### Creating Studies

Studies are automatically created when you run optimization with a study name:

```bash
python -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-name "Momentum_Unfiltered" \
  --study-name "momentum_v1" \
  --optuna-trials 100
```

### Resuming Studies

Use the same study name to continue a previous optimization:

```bash
# Continue the previous study with additional trials
python -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-name "Momentum_Unfiltered" \
  --study-name "momentum_v1" \
  --optuna-trials 200  # Will run 100 more trials
```

### Loading Best Parameters

Load the best parameters from a completed study for backtesting:

```bash
python -m src.portfolio_backtester.backtester \
  --mode backtest \
  --scenario-name "Momentum_Unfiltered" \
  --study-name "momentum_v1"  # Uses best params from this study
```

## Database Management

### Viewing Studies

You can inspect the database using any SQLite browser or command-line tools:

```bash
# Using sqlite3 command line
sqlite3 data/optuna_studies.db ".tables"
sqlite3 data/optuna_studies.db "SELECT study_name, study_id FROM studies;"
```

### Backing Up Studies

```bash
# Create a backup of your optimization studies
cp data/optuna_studies.db data/optuna_studies_backup_$(date +%Y%m%d).db
```

### Cleaning Up Studies

```bash
# Remove the database to start fresh (WARNING: This deletes all studies)
rm data/optuna_studies.db
```

### Database Size Management

The database grows with each trial. For large optimization runs:

- **Typical Size:** 100-500 KB for small studies (100-1000 trials)
- **Large Studies:** Can reach several MB for studies with 10,000+ trials
- **Cleanup:** Consider removing old studies periodically if disk space is a concern

## Best Practices

### Study Naming

Use descriptive study names that include:
- Strategy type
- Version/iteration
- Key parameters being optimized

Examples:
- `momentum_portfolio_v1`
- `calmar_strategy_robustness_test`
- `vams_optimization_2024q1`

### Study Organization

- Use consistent naming conventions
- Document study purposes and results
- Keep a log of study configurations and outcomes
- Archive completed studies before major parameter changes

### Performance Considerations

- **Parallel Optimization:** Use `--n-jobs` for faster optimization
- **Trial Pruning:** Enable `--pruning-enabled` to stop unpromising trials early
- **Study Persistence:** Studies are automatically saved after each trial

## Troubleshooting

### Database Locked Errors

If you encounter database locked errors:

```bash
# Check for running optimization processes
ps aux | grep backtester

# Wait for processes to complete or kill them if necessary
# Then retry your optimization
```

### Corrupted Database

If the database becomes corrupted:

```bash
# Remove the corrupted database
rm data/optuna_studies.db

# Restart your optimization (studies will be recreated)
```

### Study Not Found Errors

If you get "Study not found" errors:

1. Check the study name spelling
2. Verify the database exists: `ls -la data/optuna_studies.db`
3. List existing studies: `sqlite3 data/optuna_studies.db "SELECT study_name FROM studies;"`

## Configuration

The database location is configured in `src/portfolio_backtester/constants.py`:

```python
DEFAULT_OPTUNA_STORAGE_URL = "sqlite:///data/optuna_studies.db"
```

This can be customized if needed, but the default location keeps all optimization data organized in the `data/` directory.