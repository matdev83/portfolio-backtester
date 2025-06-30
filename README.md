# Portfolio Backtester

This project is a Python-based tool for backtesting portfolio strategies.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

2.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

## Usage

The main backtesting script can be run directly:

```bash
python src/portfolio_backtester/backtester.py
```

You can also specify which portfolios to backtest using the `--portfolios` argument. This allows you to select specific strategies to compare. The 'Unfiltered' portfolio and the benchmark (SPX) will always be included in the results, with the benchmark appearing as the rightmost column in the statistics table.

Example:
```bash
python src/portfolio_backtester/backtester.py --portfolios "Momentum,Sharpe Momentum"
```

The tool for downloading SPY holdings can be run with:

```bash
python src/portfolio_backtester/spy_holdings.py --out spy_holdings.csv
```

## Development Practices and Standards

To ensure the long-term quality, maintainability, and scalability of this project, all contributors are expected to adhere to the following development practices and principles:

### Modular, Layered Architecture
The project follows a modular and layered architecture. This approach promotes separation of concerns and allows for proper code re-use. Each component should have a single, well-defined responsibility and interact with other components through clear interfaces.

### Test-Driven Development (TDD)
We practice Test-Driven Development. This means that for any new feature or bug fix, a test should be written *before* the implementation code. The development cycle is as follows:
1.  **Red:** Write a failing test that captures the requirements of the new feature.
2.  **Green:** Write the simplest possible code to make the test pass.
3.  **Refactor:** Clean up and optimize the code while ensuring all tests still pass.

### SOLID Principles
We adhere to the SOLID principles of object-oriented design:

*   **S - Single-responsibility Principle:** A class should have only one reason to change, meaning it should have only one job or responsibility.
*   **O - Open-closed Principle:** Software entities (classes, modules, functions, etc.) should be open for extension but closed for modification. This means you should be able to add new functionality without changing existing code.
*   **L - Liskov Substitution Principle:** Subtypes must be substitutable for their base types. In other words, objects of a superclass should be replaceable with objects of a subclass without affecting the correctness of the program.
*   **I - Interface Segregation Principle:** No client should be forced to depend on methods it does not use. This principle suggests that larger interfaces should be split into smaller, more specific ones.
*   **D - Dependency Inversion Principle:** High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions.

### KISS (Keep It Simple, Stupid)
We favor simplicity in our designs and implementations. Avoid unnecessary complexity and over-engineering. A simple, clear solution is always preferable to a complex one, as it is easier to understand, maintain, and debug.

### Convention over Configuration
The project prefers convention over configuration. This means we rely on established conventions to reduce the number of decisions a developer needs to make. Defaults should be sane, logical, and work out-of-the-box for the most common use cases, while still allowing for configuration when necessary.
