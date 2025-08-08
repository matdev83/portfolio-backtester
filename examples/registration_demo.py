#!/usr/bin/env python3
"""
Demo showing how to use the Registration module components.

This example demonstrates the Single Responsibility Principle (SRP) design
of the registration system with three distinct classes:
- RegistrationManager: handles registration operations
- RegistrationValidator: validates registration data  
- RegistryLister: queries and lists registered components
"""

import logging

# Configure logging to see the registration activity
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_backtester.registration import (
    RegistrationManager,
    RegistrationValidator,
    RegistryLister,
)


class SampleStrategy:
    """Sample strategy class for demonstration."""
    
    def __init__(self, name: str):
        self.name = name
    
    def execute(self) -> str:
        return f"Executing strategy: {self.name}"


class SampleDataSource:
    """Sample data source class for demonstration."""
    
    def __init__(self, source_type: str):
        self.source_type = source_type
    
    def fetch_data(self) -> str:
        return f"Getting data from {self.source_type}"


def main():
    """Demonstrate the registration system."""
    print("=== Registration System Demo ===\n")
    
    # 1. Create a validator with custom rules
    print("1. Creating validator with custom rules...")
    validator = RegistrationValidator(
        reserved_names={"system", "admin", "root"},
        naming_pattern=r'^[a-z][a-z0-9_]*$',  # Must start with lowercase letter
        max_alias_count=5
    )
    
    # 2. Create registration manager with the validator
    print("2. Creating registration manager...")
    manager = RegistrationManager(validator)
    
    # 3. Create registry lister for querying
    print("3. Creating registry lister...")
    lister = RegistryLister(manager)
    
    print("\n=== Registering Components ===\n")
    
    # 4. Register some valid components
    print("4. Registering valid components...")
    
    # Register strategies
    momentum_strategy = SampleStrategy("Momentum")
    manager.register(
        "momentum_strategy",
        momentum_strategy,
        aliases=["momentum", "mom_strat"],
        metadata={"type": "strategy", "version": "1.0", "risk_level": "medium"}
    )
    
    mean_revert_strategy = SampleStrategy("Mean Reversion")
    manager.register(
        "mean_reversion_strategy",
        mean_revert_strategy,
        aliases=["mean_revert", "mr_strat"],
        metadata={"type": "strategy", "version": "2.1", "risk_level": "low"}
    )
    
    # Register data sources
    yahoo_source = SampleDataSource("Yahoo Finance")
    manager.register(
        "yahoo_finance",
        yahoo_source,
        aliases=["yahoo", "yfinance"],
        metadata={"type": "data_source", "free": True, "rate_limited": True}
    )
    
    alpha_vantage_source = SampleDataSource("Alpha Vantage")
    manager.register(
        "alpha_vantage",
        alpha_vantage_source,
        aliases=["alphav", "av"],
        metadata={"type": "data_source", "free": False, "rate_limited": True}
    )
    
    print("✓ Components registered successfully\n")
    
    # 5. Try to register invalid components
    print("5. Testing validation by attempting invalid registrations...")
    
    try:
        # Invalid name (starts with uppercase)
        manager.register("Invalid_Name", SampleStrategy("Invalid"))
    except ValueError as e:
        print(f"✓ Validation caught invalid name: {e}")
    
    try:
        # Reserved name
        manager.register("system", SampleStrategy("System"))
    except ValueError as e:
        print(f"✓ Validation caught reserved name: {e}")
    
    try:
        # Duplicate registration
        manager.register("momentum_strategy", SampleStrategy("Duplicate"))
    except ValueError as e:
        print(f"✓ Validation caught duplicate registration: {e}")
    
    print("\n=== Querying Registry ===\n")
    
    # 6. Use the lister to explore the registry
    print("6. Listing all registered components:")
    components = lister.list_components()
    for comp in components:
        print(f"  - {comp}")
    
    print("\nComponents with aliases:")
    components_with_aliases = lister.list_components(include_aliases=True)
    aliases_only = set(components_with_aliases) - set(components)
    for alias in sorted(aliases_only):
        print(f"  - {alias} (alias)")
    
    # 7. Get detailed component information
    print("\n7. Getting detailed component information:")
    info = lister.get_component_info("momentum_strategy")
    if info:
        print(f"Component: {info['name']}")
        print(f"Type: {info['component_type']}")
        print(f"Aliases: {', '.join(info['aliases'])}")
        print(f"Metadata: {info['metadata']}")
    
    # Test alias resolution
    print("\n8. Testing alias resolution:")
    alias_info = lister.get_component_info("momentum")
    if alias_info:
        print(f"Alias 'momentum' resolves to: {alias_info['name']}")
        print(f"Is alias: {alias_info['is_alias']}")
    
    # 9. Filter components by type and metadata
    print("\n9. Filtering components:")
    
    # Filter by type
    strategy_components = lister.filter_by_type(SampleStrategy)
    print(f"Strategy components: {strategy_components}")
    
    data_source_components = lister.filter_by_type(SampleDataSource)
    print(f"Data source components: {data_source_components}")
    
    # Filter by metadata
    free_sources = lister.filter_by_metadata(free=True)
    print(f"Free data sources: {free_sources}")
    
    high_risk_strategies = lister.filter_by_metadata(risk_level="high")
    print(f"High risk strategies: {high_risk_strategies}")
    
    # 10. Search components
    print("\n10. Searching components:")
    strategy_matches = lister.search_components("*strategy*")
    print(f"Components matching '*strategy*': {strategy_matches}")
    
    # 11. Get registry summary
    print("\n11. Registry summary:")
    summary = lister.get_registry_summary()
    print(f"Total components: {summary['statistics']['total_components']}")
    print(f"Total aliases: {summary['statistics']['total_aliases']}")
    print(f"Type distribution: {summary['type_distribution']}")
    
    # 12. Check registry integrity
    print("\n12. Checking registry integrity:")
    issues = lister.validate_registry_integrity()
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Registry integrity check passed")
    
    # 13. Demonstrate component usage
    print("\n=== Using Registered Components ===\n")
    
    print("13. Using registered components:")
    
    # Get and use a strategy
    strategy = manager.get_component("momentum")  # Using alias
    if strategy:
        print(f"Retrieved strategy: {strategy}")
        print(f"Execution result: {strategy.execute()}")
    
    # Get and use a data source
    data_source = manager.get_component("yahoo_finance")
    if data_source:
        print(f"Retrieved data source: {data_source}")
        print(f"Data fetch result: {data_source.fetch_data()}")
    
    # 14. Clean up
    print("\n14. Cleaning up - deregistering a component:")
    success = manager.deregister("alpha_vantage")
    print(f"Deregistration successful: {success}")
    
    print(f"Components after cleanup: {lister.list_components()}")
    
    print("\n=== Demo Complete ===")
    print("\nThis demo showed how the three registration classes work together:")
    print("- RegistrationManager: Handled all registration/deregistration operations")
    print("- RegistrationValidator: Enforced naming rules and data validation")
    print("- RegistryLister: Provided comprehensive querying and inspection capabilities")
    print("\nEach class has a single, well-defined responsibility (SRP)")
    print("and they collaborate through composition rather than inheritance.")


if __name__ == "__main__":
    main()
