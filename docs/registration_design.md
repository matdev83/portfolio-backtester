# Registration Module Design

## Overview

This document describes the design of the registration module, which implements a SOLID-compliant component registration system following the Single Responsibility Principle (SRP). The system consists of three main classes that collaborate through composition to provide comprehensive registration management capabilities.

## Architecture

### Class Responsibilities

The registration system is designed around three distinct classes, each with a single, well-defined responsibility:

#### 1. RegistrationManager
**Responsibility**: Manages the lifecycle of component registrations

**Core Functions**:
- Register components with unique names
- Deregister components
- Retrieve registered components by name or alias
- Manage component metadata
- Handle aliases and name resolution
- Ensure registration integrity

**Key Methods**:
- `register(name, component, *, aliases=None, metadata=None, force=False)`
- `deregister(name) -> bool`
- `get_component(name) -> Optional[Any]`
- `is_registered(name) -> bool`
- `get_metadata(name) -> Optional[Dict[str, Any]]`
- `update_metadata(name, metadata) -> bool`

#### 2. RegistrationValidator
**Responsibility**: Validates registration data and enforces registration rules

**Core Functions**:
- Validate registration data structure and types
- Enforce naming conventions and constraints
- Validate component metadata
- Check business rules for registration
- Provide detailed validation error messages
- Support custom validation functions

**Key Methods**:
- `validate(data) -> List[str]`
- `add_custom_validator(validator)`
- `set_naming_pattern(pattern)`
- `add_reserved_name(name)`
- `remove_reserved_name(name) -> bool`

**Validation Features**:
- Name format validation (regex patterns)
- Reserved name checking
- Alias validation and uniqueness
- Metadata structure validation
- Custom validation functions
- Configurable constraints

#### 3. RegistryLister
**Responsibility**: Queries and lists registered components (read-only operations)

**Core Functions**:
- List all registered components
- Filter components by various criteria
- Search components by name patterns
- Provide component information and metadata
- Generate registry reports and statistics
- Validate registry integrity

**Key Methods**:
- `list_components(*, include_aliases=False) -> List[str]`
- `get_component_info(name) -> Optional[Dict[str, Any]]`
- `search_components(pattern, *, case_sensitive=False) -> List[str]`
- `filter_components(criterion) -> List[str]`
- `filter_by_type(component_type) -> List[str]`
- `filter_by_metadata(**criteria) -> List[str]`
- `get_registry_summary() -> Dict[str, Any]`
- `validate_registry_integrity() -> List[str]`

### Design Principles

#### Single Responsibility Principle (SRP)
Each class has a single, focused responsibility:
- **RegistrationManager**: Registration lifecycle management
- **RegistrationValidator**: Data validation and rule enforcement
- **RegistryLister**: Registry querying and inspection

#### Composition over Inheritance
The classes collaborate through composition rather than inheritance:
- RegistrationManager uses RegistrationValidator for validation
- RegistryLister accesses RegistrationManager's registry data
- No complex inheritance hierarchies

#### Open/Closed Principle
The system is open for extension but closed for modification:
- Custom validators can be added to RegistrationValidator
- New filtering criteria can be added to RegistryLister
- The core interfaces remain stable

#### Dependency Inversion
- RegistrationManager accepts a validator as a dependency
- RegistryLister accepts a registration manager as a dependency
- Dependencies are injected, not hardcoded

## Key Features

### Advanced Validation
- **Naming Patterns**: Regex-based name validation
- **Reserved Names**: Configurable list of forbidden names
- **Alias Management**: Comprehensive alias validation and conflict detection
- **Metadata Validation**: Structured metadata validation with type checking
- **Custom Validators**: Extensible validation system

### Comprehensive Querying
- **Pattern Search**: Wildcard-based component searching
- **Type Filtering**: Filter components by their Python type
- **Metadata Filtering**: Advanced metadata-based filtering with operators
- **Similarity Search**: Find components with similar names
- **Registry Statistics**: Comprehensive registry analytics

### Robust Management
- **Alias Support**: Multiple aliases per component with resolution
- **Metadata Management**: Rich metadata support with update capabilities
- **Force Override**: Safe component replacement with explicit confirmation
- **Integrity Checking**: Built-in registry consistency validation
- **Error Handling**: Comprehensive error reporting and logging

### Integration Features
- **Logging**: Comprehensive logging for debugging and auditing
- **Type Safety**: Full type hints for better IDE support and reliability
- **Testing**: Extensive test coverage with isolated unit tests
- **Documentation**: Complete docstrings with Google style formatting

## Usage Examples

### Basic Usage
```python
from portfolio_backtester.registration import (
    RegistrationManager,
    RegistrationValidator,
    RegistryLister,
)

# Create validator with custom rules
validator = RegistrationValidator(
    reserved_names={"system", "admin"},
    naming_pattern=r'^[a-z][a-z0-9_]*$',
    max_alias_count=5
)

# Create manager with validator
manager = RegistrationManager(validator)

# Create lister for querying
lister = RegistryLister(manager)

# Register a component
manager.register(
    "my_strategy",
    MyStrategy(),
    aliases=["strategy", "strat"],
    metadata={"version": "1.0", "type": "strategy"}
)

# Query components
strategies = lister.filter_by_metadata(type="strategy")
info = lister.get_component_info("my_strategy")
```

### Advanced Filtering
```python
# Filter by type
strategy_components = lister.filter_by_type(MyStrategy)

# Filter by metadata with operators
free_sources = lister.filter_by_metadata(
    free=True,
    version={"$regex": r"^2\."}  # Version starts with "2."
)

# Search with patterns
matching = lister.search_components("*strategy*", case_sensitive=False)

# Get similarity suggestions
similar = lister.find_similar_names("strategi", max_results=3)
```

### Custom Validation
```python
def validate_version_format(data):
    """Custom validator for version format."""
    errors = []
    metadata = data.get("metadata", {})
    version = metadata.get("version")
    
    if version and not re.match(r'^\d+\.\d+(\.\d+)?$', version):
        errors.append("Version must be in X.Y or X.Y.Z format")
    
    return errors

# Add custom validator
validator.add_custom_validator(validate_version_format)
```

## Benefits

### For Developers
- **Clear Separation**: Each class has a well-defined purpose
- **Easy Testing**: Classes can be tested in isolation
- **Maintainable**: Changes to one class don't affect others
- **Extensible**: New functionality can be added without breaking existing code

### For Users
- **Robust Validation**: Comprehensive data validation prevents errors
- **Flexible Querying**: Multiple ways to find and filter components
- **Rich Metadata**: Detailed component information and statistics
- **Reliable Operation**: Built-in integrity checking and error handling

### For the System
- **Performance**: Efficient O(1) lookups with caching
- **Scalability**: Handles large numbers of registered components
- **Memory Efficient**: Minimal memory overhead
- **Thread Safe**: Can be made thread-safe with minimal changes

## Integration with Existing Codebase

The registration module is designed to integrate seamlessly with the existing portfolio backtester codebase:

- **Consistent Patterns**: Follows existing code patterns and conventions
- **Compatible APIs**: Uses similar method signatures and return types
- **Error Handling**: Consistent exception handling and logging
- **Testing Strategy**: Matches existing test organization and style

The module can be used to register and manage various system components such as:
- Strategy classes
- Data source implementations
- Custom timing controllers
- Performance analyzers
- Report generators

## Future Enhancements

Potential future enhancements that maintain the SRP design:

1. **Persistence**: Add optional persistence for registry state
2. **Events**: Add event system for registration/deregistration notifications
3. **Namespaces**: Support hierarchical component namespaces
4. **Versioning**: Enhanced version management and compatibility checking
5. **Security**: Access control and permission-based operations
6. **Clustering**: Distributed registry for multi-node deployments

Each enhancement would be implemented as separate, focused components that integrate with the existing classes through composition.
