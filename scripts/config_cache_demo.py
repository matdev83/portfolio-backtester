#!/usr/bin/env python3
"""
Configuration Cache Management Demo

This script demonstrates the intelligent caching functionality for YAML configuration validation.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from portfolio_backtester.config_cache import get_cache


def main():
    print("=== Configuration Cache Management Demo ===\n")

    cache = get_cache()

    # Show current cache info
    print("1. Current Cache Status:")
    info = cache.get_cache_info()
    print(f"   Cache file: {info['cache_file']}")
    print(f"   Cache exists: {info['cache_exists']}")
    print(f"   Cached keys: {info['cached_keys']}")

    if info["cached_keys"]:
        for key in info["cached_keys"]:
            timestamp_key = f"{key}_timestamp"
            counts_key = f"{key}_file_counts"
            if timestamp_key in info and counts_key in info:
                import datetime

                dt = datetime.datetime.fromtimestamp(info[timestamp_key])
                print(f"   {key}:")
                print(f"     - Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     - File counts: {info[counts_key]}")

    print("\n2. Testing Cache Validation:")

    # Test strategy validation cache
    strategy_cache_valid, reason = cache.is_cache_valid("strategy_validation_comprehensive")
    print(f"   Strategy validation cache valid: {strategy_cache_valid}")
    if reason:
        print(f"   Reason: {reason}")

    # Test full YAML validation cache
    yaml_cache_valid, reason = cache.is_cache_valid("yaml_validation_full")
    print(f"   YAML validation cache valid: {yaml_cache_valid}")
    if reason:
        print(f"   Reason: {reason}")

    print("\n3. Monitored Directories:")
    for name, path in info["monitored_directories"].items():
        print(f"   {name}: {path}")

    print("\n4. Available Cache Operations:")
    print("   - Clear specific cache: cache.clear_cache('cache_key')")
    print("   - Clear all cache: cache.clear_cache()")
    print("   - Check cache validity: cache.is_cache_valid('cache_key')")
    print("   - Get cached result: cache.get_cached_result('cache_key')")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
