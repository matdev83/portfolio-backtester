# Mini PRD: YAML Configuration Files Validation Process

## 1. What is it?
The YAML Configuration Files Validation Process is a system designed to ensure the integrity and correctness of YAML configuration files used by the application. It performs checks at two key stages: application startup and during the user-initiated optimization process.

## 2. Why does it exist?
To prevent runtime errors, misconfigurations, and unexpected behavior by validating configuration files against expected schemas, strategy implementations, and code capabilities. It ensures that:
*   The application is in a known-good state at startup.
*   Optimization runs are performed with valid and complete configurations, reducing the likelihood of failures mid-process and improving user confidence.
*   Code organization is maintained by ensuring Python files within a strategy's directory are relevant to that strategy.

## 3. How does it work?

### 3.1. Application Startup Strategy & Default Config Validation
*   **Trigger:** On application load.
*   **Process:**
    1.  The application enumerates available concrete strategy classes via the registry-backed discovery engine. Discovery roots now include `strategies/builtins/{portfolio,signal,meta}` and `strategies/user/{portfolio,signal,meta}`. `strategies/examples/**` are explicitly excluded.
        *   The registry walks modules under the above roots and traverses the `BaseStrategy` hierarchy to find all concrete (non-abstract) subclasses.
        *   Class names (e.g., `DummyStrategyForTesting`) are mapped to canonical identifiers (e.g., `dummy_strategy_for_testing`) and known aliases.
    2.  For each discovered strategy identifier, it checks for the existence of a corresponding `default.yaml` configuration file. These files reside under mirrored locations:
        *   Built-ins: `config/scenarios/builtins/<category>/<strategy>/default.yaml`
        *   User:     `config/scenarios/user/<category>/<strategy>/default.yaml`
       Example: `config/scenarios/builtins/signal/dummy_signal_strategy/default.yaml`.
    3.  Python File Consistency Check: For each discovered strategy, the application inspects its implementation module path under `strategies/{builtins|user}/...`. Any stray Python files colocated with a strategy but not matching the strategy identifier may be logged as warnings to encourage tidy layout.
    4.  If a `default.yaml` file is **not found** for a discovered strategy in either `builtins` or `user` trees, the application logs an error, reports the missing configuration for that specific strategy, and **refuses to start**.
*   **Outcome:** Ensures that every strategy the application is aware of has a baseline configuration file and that its associated Python implementation files are logically organized, preventing scenarios where a strategy might be called but lacks defined parameters or where unrelated code files exist within a strategy's directory.

### 3.2. Deep Configuration File Validator (Optimization Pre-Flight Check)
*   **Trigger:** Executed when a user initiates an optimization for a specific scenario (i.e., a YAML file under `config/scenarios/{builtins|user}/<category>/<strategy>/*.yaml`).
*   **Process:** A two-step deep validation is performed:

    **a) Validation of the Strategy's `default.yaml` (in builtins or user tree):**
    1.  **Syntax Check:** Verifies that the `default.yaml` file is well-formed YAML.
    2.  **Required Parameters Check:** Ensures all parameters required by the strategy are present in the `default.yaml`.
    3.  **Unexpected Parameters Check:** Flags any parameters present in the `default.yaml` that are not recognized or handled by the strategy's code.
    4.  **Parameter Formal Validity:** Validates the format and value ranges of individual parameters (e.g., numeric ranges, string formats, enum values).
    5.  **Optimizable Variables Introspection:** Performs code introspection on the strategy to identify all parameters designated as optimizable. It then checks if these optimizable parameters are indeed present in the `default.yaml` file.
    6.  **Parameter Type Validation:** Validates that the data types of parameters in the `default.yaml` match the expected types defined in the strategy (e.g., an integer parameter is not provided as a string).

    **b) Validation of the Selected Scenario (`<scenario_filename>`):**
    1.  The same comprehensive validation steps applied to the `default.yaml` in step (a) are applied to the user's specific scenario YAML file being run.
    2.  This includes syntax, required/parameter checks, formal validity, introspection for optimizable variables, and type validation.
    3.  The validation ensures that the scenario file correctly inherits or overrides parameters from the strategy's `default.yaml` and that any parameters specific to the scenario are valid.

*   **Outcome:** Provides a robust pre-flight check before computationally expensive optimization runs, ensuring that both the base strategy configuration and the user's specific scenario are correct and complete, thus preventing optimization failures due to configuration errors.

## 4. Success Criteria / Outcomes
*   The application fails to start with a clear error message if any strategy is missing its `default.yaml`.
*   Warnings are logged if Python files are found within a strategy's directory that do not contain the strategy's name, promoting better code organization.
*   Users attempting to run an optimization with an invalid configuration file receive detailed, specific error messages highlighting the exact nature of the validation failure (e.g., "Parameter 'lookback_period' is missing and required," "Parameter 'risk_factor' is not a recognized option for this strategy," "Type mismatch for parameter 'threshold': expected float, got string").
*   Optimization runs are consistently initiated only with valid and complete configurations.
*   The system provides clear feedback to users, enabling them to quickly identify and fix configuration issues.
