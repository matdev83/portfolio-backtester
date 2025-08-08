# Configuration Validation Caching System

## Overview

A comprehensive caching system has been implemented for YAML configuration validation to avoid redundant analysis when files haven't changed between runs. This significantly improves performance for repeated validation operations.

## Features

### Two-Stage Cache Invalidation
The system uses a sophisticated two-stage cache invalidation approach:

1. **Stage 1 - File Count Check**: Quick validation that the number of files hasn't changed
2. **Stage 2 - Timestamp Check**: Detailed validation that no individual files have been modified

### Monitored Locations
The cache monitors these critical directories and files:
- `config/scenarios/` (all YAML files recursively)
- `src/portfolio_backtester/strategies/` (all Python files recursively)
- `config/parameters.yaml` (single file)

### Cache Storage
- Cache is stored persistently in `data/config_validation_cache.json`
- Survives across application runs
- Automatically created on first use

## Usage

### Command Line Interface

#### Basic Validation (with automatic caching)
```bash
python -m src.portfolio_backtester.config_loader --validate
```

#### Force Refresh (ignore cache)
```bash
python -m src.portfolio_backtester.config_loader --validate --force-refresh
```

#### Cache Management
```bash
# Show cache information
python -m src.portfolio_backtester.config_loader --cache-info

# Clear entire cache
python -m src.portfolio_backtester.config_loader --clear-cache
```

### Programmatic Usage

```python
from portfolio_backtester.config_cache import get_cache

# Get cache instance
cache = get_cache()

# Check if cache is valid
is_valid, reason = cache.is_cache_valid("validation_key")

# Get cached result
result = cache.get_cached_result("validation_key")

# Store new result
cache.store_result({"data": "value"}, "validation_key")

# Clear cache
cache.clear_cache()  # Clear all
cache.clear_cache("specific_key")  # Clear specific key
```

## Cache Keys

The system uses different cache keys for different validation types:

- `strategy_validation_comprehensive`: Strategy configuration cross-validation results
- `yaml_validation_full`: Complete YAML file validation results

## Performance Benefits

### Before Caching
- Every validation run required full file system scan
- All YAML files parsed and validated on each run
- Strategy cross-reference validation performed every time
- Typical validation time: 2-5 seconds for large config sets

### After Caching
- Subsequent validations use cached results when files unchanged
- Cache validation typically completes in <100ms
- Only performs full validation when files actually change
- Dramatic improvement for development workflows

## Cache Invalidation Triggers

The cache is automatically invalidated when:

1. **File Count Changes**:
   - New YAML files added to scenarios
   - YAML files removed from scenarios
   - New Python strategy files added
   - Python strategy files removed
   - Parameters file created/deleted

2. **File Modifications**:
   - Any YAML file in scenarios modified (timestamp/size change)
   - Any Python file in strategies modified (timestamp/size change)
   - Parameters file modified (timestamp/size change)

## Informative Messages

The system provides clear messages about cache usage:

```
Using cached validation results...
[OK] All configuration files are valid (cached)
```

```
Running fresh validation: File modifications detected: scenarios: 2 modified files
```

```
Running fresh validation: File count changes detected: scenarios: 24 -> 25 files
```

## Integration Points

### Automatic Integration
The caching system is automatically integrated into:
- `config_loader.py` - Main configuration loading
- `strategy_config_validator.py` - Strategy validation
- `validate_config_files()` - Standalone validation function

### Framework Integration
- Cache validation runs during framework startup
- Cross-check validation includes caching
- No changes required to existing code

## Cache Management Tools

### Demo Script
```bash
python scripts/config_cache_demo.py
```

### Cache Information Display
Shows:
- Cache file location
- Cache existence status
- Cached validation keys
- Timestamps and file counts
- Monitored directories

## Benefits for Development

1. **Faster Development Cycles**: Repeated validation during development is nearly instantaneous
2. **Efficient CI/CD**: Build processes skip redundant validation when configs unchanged
3. **Resource Conservation**: Reduces unnecessary file I/O and processing
4. **Improved User Experience**: Immediate feedback when no changes detected

## Technical Implementation

### Cache Structure
```json
{
  "validation_key": {
    "result": {
      "is_valid": true,
      "errors": []
    },
    "state": {
      "timestamp": 1704628347.123,
      "file_counts": {
        "scenarios": 25,
        "strategies": 56,
        "parameters": 1
      },
      "files": {
        "scenarios": [
          {"path": "...", "mtime": 1704628347.123, "size": 1234},
          ...
        ],
        "strategies": [...],
        "parameters": [...]
      }
    }
  }
}
```

### File Monitoring
- Uses `os.stat()` for file metadata
- Tracks modification time (`mtime`) and file size
- Recursive directory scanning with `Path.rglob()`
- Efficient comparison algorithms

## Error Handling

The cache system is designed to be robust:
- Cache corruption is handled gracefully (rebuilds cache)
- File system errors don't break validation
- Cache failures fall back to fresh validation
- Comprehensive logging for debugging

## Future Enhancements

Potential improvements for future versions:
- Selective cache invalidation (only affected validation types)
- Cache compression for large projects
- Distributed cache for team environments
- Cache analytics and performance metrics
