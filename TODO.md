# Strategy YAML Checker Improvements TODO List

This file tracks progress on enhancing the strategy YAML file checker for universal and per-strategy parameter validation.

## TODO Items

1. **Gather More Context**  
   - [x] Read an example strategy file (e.g., dummy_strategy.py) to confirm tunable params structure.  
   - [x] Search for all strategies implementing tunable_parameters() to assess update scope.  

2. **Enhance Per-Strategy Introspection**  
   - [x] Update base_strategy.py to return richer metadata (e.g., dict with type/range).  
   - [x] Update scenario_validator.py to use this metadata for deeper checks.  
   - [x] Update individual strategy files (one-by-one) to provide metadata.  

3. **Improve Universal Checks**  
   - [x] Add value constraints and logical checks to scenario_validator.py.  

4. **Add Tests and Verify**  
   - [x] Create/edit unit tests for new validation logic.  
   - [x] Run full test suite and end-to-end CLI backtest to confirm no regressions.  

5. **Cleanup and Document**  
   - [x] Remove any temp/debug code.  
   - [x] Update README.md if needed.  
   - [x] Final verification run.

Task complete: All enhancements implemented, tests passing, end-to-end verified.

*Last updated: [Current Date]* 