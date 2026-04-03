# Implementation Plan

## LLM Agents Note: Actions AFTER Each File Edit

After each completed file edit, agents MUST run the following quality-assurance commands against the single modified file (in this order):

```bash
./.venv/Scripts/python.exe -m black <modified_filename>
./.venv/Scripts/python.exe -m ruff check --fix <modified_filename>
./.venv/Scripts/python.exe -m mypy <modified_filename>
```

Notes:

- Always use the Windows venv interpreter path shown above.
- Replace `<modified_filename>` with the exact path to the changed file.
- Run these before proceeding to additional edits or committing.

## Task List

- [x] 1. Fix Signal-Based Timing UnboundLocalError
  - Identify the exact location where `base_scan_dates` is referenced before assignment
  - Add proper variable initialization before the conditional logic
  - Test the fix with existing timing tests
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Fix Momentum Strategy Pandas Index Ambiguity
  - Locate all instances of direct pandas Index boolean evaluation in momentum strategies
  - Replace ambiguous boolean checks with explicit pandas methods (.empty, .any(), .all())
  - Ensure RoRo signal processing handles Index objects correctly
  - Test momentum strategy signal generation with various input types
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. Fix Trade Aggregator Negative Quantity Validation
  - Analyze the trade creation and validation logic to understand quantity/side relationship
  - Modify validation to allow negative quantities for sell trades when appropriate
  - Ensure validation still catches truly invalid trade configurations
  - Update trade aggregator tests to cover both buy and sell scenarios
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. Fix Position Sizer Method Signatures
  - Standardize all position sizer `calculate_weights` methods to accept `(signals, prices, **kwargs)`
  - Update EqualWeightSizer and any other inconsistent implementations
  - Ensure backward compatibility where possible
  - Update position sizer tests to use the correct method signature
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 5. Fix Optuna Study Name Conflicts
  - Implement unique study name generation using timestamps or UUIDs
  - Update all optimization tests to use unique study names
  - Add cleanup logic to remove test studies after completion
  - Ensure tests can run in parallel without conflicts
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 6. Fix Universe Configuration File Dependencies
  - Create proper test fixtures for universe configuration tests
  - Implement mocking for file system operations where appropriate
  - Update tests to either create required files or adapt expectations to existing files
  - Ensure universe tests are isolated and don't depend on external file state
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 7. Run comprehensive test validation
  - Execute the full test suite to verify all fixes work correctly
  - Identify any remaining issues that may have been uncovered by the fixes
  - Ensure no regressions were introduced by the changes
  - Document the impact of each fix on test pass rates
  - _Requirements: All requirements validation_

- [ ] 8. Create regression prevention tests

  - Add specific unit tests for each fixed bug to prevent future regressions
  - Implement property-based tests where applicable
  - Add integration tests to verify component interactions still work correctly
  - Update CI/CD pipeline to catch these specific issues in the future
  - _Requirements: All requirements - regression prevention_