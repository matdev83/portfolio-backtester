# API Deprecation Guide

This guide explains how to use the deprecation decorators provided by the API stability protection system to manage breaking changes gracefully.

## Overview

The API stability protection system provides two main decorators for handling deprecation:

1. `@deprecated` - For marking entire methods as deprecated
2. `@deprecated_signature` - For handling parameter name changes while maintaining backward compatibility

## @deprecated Decorator

Use this decorator when you want to mark an entire method as deprecated and guide users to a replacement.

### Basic Usage

```python
from src.portfolio_backtester.api_stability import deprecated

@deprecated(reason="This method has been replaced with a more efficient implementation")
def old_method(data):
    # Legacy implementation
    return process_data_old_way(data)

def new_method(data, use_cache=True):
    # New and improved implementation
    return process_data_new_way(data, use_cache)
```

### Full Usage with All Parameters

```python
@deprecated(
    reason="Better implementation available with improved performance",
    version="2.0",
    removal_version="3.0", 
    migration_guide="Use new_method() instead, which provides caching and better error handling"
)
def old_method(data):
    return process_data_old_way(data)
```

### Parameters

- `reason` (str): Explanation of why the method is deprecated
- `version` (str, optional): Version in which the method was deprecated
- `removal_version` (str, optional): Version in which the method will be removed
- `migration_guide` (str, optional): Detailed instructions for migrating to new method
- `category` (str, optional): Warning category (default: "DeprecationWarning")

## @deprecated_signature Decorator

Use this decorator when you need to change parameter names but want to maintain backward compatibility temporarily.

### Basic Usage

```python
from src.portfolio_backtester.api_stability import deprecated_signature

@deprecated_signature(
    old_signature="set_parameters(period, threshold, format='json')",
    new_signature="set_parameters(period, threshold, output_format='json')",
    parameter_mapping={"format": "output_format"}
)
def set_parameters(self, period, threshold, output_format='json'):
    return {"period": period, "threshold": threshold, "format": output_format}
```

### Multiple Parameter Changes

```python
@deprecated_signature(
    old_signature="backtest(data, start, end, verbose=False, format='dict')",
    new_signature="backtest(data, start, end, show_progress=False, output_format='dict')",
    version="2.2",
    removal_version="3.0",
    parameter_mapping={
        "verbose": "show_progress", 
        "format": "output_format"
    }
)
def backtest(self, data, start, end, show_progress=False, output_format='dict'):
    # Implementation using new parameter names
    return run_backtest(data, start, end, show_progress, output_format)
```

### Parameters

- `old_signature` (str): Description of the old method signature
- `new_signature` (str): Description of the new method signature
- `version` (str, optional): Version in which the signature was deprecated
- `removal_version` (str, optional): Version in which old signature support will be removed
- `parameter_mapping` (dict, optional): Dictionary mapping old parameter names to new ones

## Best Practices

### 1. Provide Clear Migration Guidance

Always include specific instructions on how to migrate:

```python
@deprecated(
    reason="Method signature changed to support new features",
    migration_guide="Replace old_method(data, format) with new_method(data, output_format=format)"
)
```

### 2. Use Version Numbers

Include version information to help users plan migrations:

```python
@deprecated(
    version="2.1",           # When it was deprecated
    removal_version="3.0"    # When it will be removed
)
```

### 3. Gradual Migration with @deprecated_signature

For parameter changes, use `@deprecated_signature` to allow both old and new parameter names:

```python
# Users can call either:
# method(data, format='json')      # Old way (shows warning)
# method(data, output_format='json')  # New way (no warning)
```

### 4. Combine with Other Decorators

Deprecation decorators work well with other API stability decorators:

```python
@deprecated(reason="Old implementation")
@api_stable(version="1.0")
def old_method(param: int) -> str:
    return str(param)
```

## Warning Output Examples

### @deprecated Warning

```
DeprecationWarning: Call to deprecated method 'ExampleClass.old_method'. 
Reason: This method has been replaced with a more efficient implementation. 
Deprecated since version 2.0. Will be removed in version 3.0. 
Migration guide: Use new_method() instead, which provides better performance.
```

### @deprecated_signature Warning

```
DeprecationWarning: Method 'ExampleClass.set_parameters' signature has changed. 
Old signature: set_parameters(period, threshold, format='json'). 
New signature: set_parameters(period, threshold, output_format='json'). 
Parameter name changes: 'format' -> 'output_format'. 
Deprecated since version 2.1. Old parameter names will be removed in version 3.0.
```

## Integration with Development Workflow

### 1. Enable Warnings in Development

```python
import warnings
warnings.simplefilter("always", DeprecationWarning)
```

### 2. Testing Deprecated Methods

The decorators preserve function metadata and can be tested normally:

```python
def test_deprecated_method():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_method("test_data")
        
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()
        assert result == expected_result
```

### 3. Monitoring Deprecation Usage

The decorators log warnings that can be monitored in production:

```python
import logging
logging.getLogger('src.portfolio_backtester.api_stability.protection').setLevel(logging.WARNING)
```

## Migration Timeline Example

Here's a typical deprecation timeline:

1. **Version 2.0**: Introduce new method alongside old method
2. **Version 2.1**: Add `@deprecated` decorator to old method
3. **Version 2.2-2.9**: Keep both methods, encourage migration
4. **Version 3.0**: Remove deprecated method

This gives users multiple versions to migrate their code gradually.