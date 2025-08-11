"""Tests for registration module components."""

import pytest
from unittest.mock import Mock

from portfolio_backtester.registration import (
    RegistrationManager,
    RegistrationValidator,
    RegistryLister,
)


class TestRegistrationManager:
    """Test RegistrationManager functionality."""

    def test_init(self):
        """Test RegistrationManager initialization."""
        manager = RegistrationManager()
        assert manager._registry == {}
        assert manager._metadata == {}
        assert manager._aliases == {}
        assert isinstance(manager._validator, RegistrationValidator)

    def test_register_component(self):
        """Test registering a component."""
        manager = RegistrationManager()
        component = Mock()

        manager.register("test_component", component)

        assert "test_component" in manager._registry
        assert manager._registry["test_component"] is component

    def test_register_with_aliases(self):
        """Test registering a component with aliases."""
        manager = RegistrationManager()
        component = Mock()

        manager.register("test_component", component, aliases=["alias1", "alias2"])

        assert manager._aliases["alias1"] == "test_component"
        assert manager._aliases["alias2"] == "test_component"

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate names raises error."""
        manager = RegistrationManager()
        component1 = Mock()
        component2 = Mock()

        manager.register("test_component", component1)

        with pytest.raises(ValueError, match="already registered"):
            manager.register("test_component", component2)

    def test_register_duplicate_with_force(self):
        """Test that force=True allows overwriting existing registration."""
        manager = RegistrationManager()
        component1 = Mock()
        component2 = Mock()

        manager.register("test_component", component1)
        manager.register("test_component", component2, force=True)

        assert manager._registry["test_component"] is component2

    def test_deregister_component(self):
        """Test deregistering a component."""
        manager = RegistrationManager()
        component = Mock()

        manager.register("test_component", component, aliases=["alias1"])
        result = manager.deregister("test_component")

        assert result is True
        assert "test_component" not in manager._registry
        assert "alias1" not in manager._aliases

    def test_deregister_nonexistent(self):
        """Test deregistering non-existent component returns False."""
        manager = RegistrationManager()
        result = manager.deregister("nonexistent")
        assert result is False

    def test_get_component(self):
        """Test retrieving a component."""
        manager = RegistrationManager()
        component = Mock()

        manager.register("test_component", component)
        retrieved = manager.get_component("test_component")

        assert retrieved is component

    def test_get_component_by_alias(self):
        """Test retrieving a component by alias."""
        manager = RegistrationManager()
        component = Mock()

        manager.register("test_component", component, aliases=["alias1"])
        retrieved = manager.get_component("alias1")

        assert retrieved is component

    def test_is_registered(self):
        """Test checking if component is registered."""
        manager = RegistrationManager()
        component = Mock()

        manager.register("test_component", component, aliases=["alias1"])

        assert manager.is_registered("test_component") is True
        assert manager.is_registered("alias1") is True
        assert manager.is_registered("nonexistent") is False


class TestRegistrationValidator:
    """Test RegistrationValidator functionality."""

    def test_init(self):
        """Test RegistrationValidator initialization."""
        validator = RegistrationValidator()
        assert validator._custom_validators == []
        assert validator._naming_pattern is None
        assert validator._reserved_names == set()
        assert validator._max_alias_count == 10

    def test_validate_valid_data(self):
        """Test validation of valid registration data."""
        validator = RegistrationValidator()
        data = {
            "name": "test_component",
            "component": Mock(),
            "aliases": ["alias1"],
            "metadata": {"version": "1.0"},
        }

        errors = validator.validate(data)
        assert errors == []

    def test_validate_missing_name(self):
        """Test validation fails when name is missing."""
        validator = RegistrationValidator()
        data = {"component": Mock()}

        errors = validator.validate(data)
        assert any("Missing required 'name' field" in error for error in errors)

    def test_validate_missing_component(self):
        """Test validation fails when component is missing."""
        validator = RegistrationValidator()
        data = {"name": "test"}

        errors = validator.validate(data)
        assert any("Missing required 'component' field" in error for error in errors)

    def test_validate_invalid_name_type(self):
        """Test validation fails for non-string name."""
        validator = RegistrationValidator()
        data = {"name": 123, "component": Mock()}

        errors = validator.validate(data)
        assert any("Name must be a string" in error for error in errors)

    def test_validate_reserved_name(self):
        """Test validation fails for reserved names."""
        validator = RegistrationValidator(reserved_names={"reserved"})
        data = {"name": "reserved", "component": Mock()}

        errors = validator.validate(data)
        assert any("is reserved and cannot be used" in error for error in errors)

    def test_validate_invalid_aliases(self):
        """Test validation of invalid aliases."""
        validator = RegistrationValidator()
        data = {
            "name": "test",
            "component": Mock(),
            "aliases": ["test", "test"],  # Duplicate alias and same as name
        }

        errors = validator.validate(data)
        assert any("cannot be the same as component name" in error for error in errors)
        assert any("Duplicate alias" in error for error in errors)


class TestRegistryLister:
    """Test RegistryLister functionality."""

    def test_init(self):
        """Test RegistryLister initialization."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)
        assert lister._manager is manager

    def test_list_components(self):
        """Test listing components."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)

        manager.register("component1", Mock())
        manager.register("component2", Mock(), aliases=["alias1"])

        names = lister.list_components()
        assert "component1" in names
        assert "component2" in names
        assert "alias1" not in names

        names_with_aliases = lister.list_components(include_aliases=True)
        assert "alias1" in names_with_aliases

    def test_get_component_info(self):
        """Test getting component information."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)
        component = Mock()

        manager.register(
            "test_component", component, aliases=["alias1"], metadata={"version": "1.0"}
        )

        info = lister.get_component_info("test_component")
        assert info is not None
        assert info["name"] == "test_component"
        assert info["component"] is component
        assert "alias1" in info["aliases"]
        assert info["metadata"]["version"] == "1.0"
        assert info["is_alias"] is False

    def test_get_component_info_by_alias(self):
        """Test getting component info by alias."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)
        component = Mock()

        manager.register("test_component", component, aliases=["alias1"])

        info = lister.get_component_info("alias1")
        assert info is not None
        assert info["name"] == "test_component"
        assert info["is_alias"] is True
        assert info["resolved_name"] == "test_component"

    def test_filter_by_type(self):
        """Test filtering components by type."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)

        str_component = "test_string"
        int_component = 42

        manager.register("string_comp", str_component)
        manager.register("int_comp", int_component)

        str_components = lister.filter_by_type(str)
        assert "string_comp" in str_components
        assert "int_comp" not in str_components

    def test_search_components(self):
        """Test searching components by pattern."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)

        manager.register("test_component", Mock())
        manager.register("production_component", Mock())
        manager.register("other_thing", Mock())

        matches = lister.search_components("*component*")
        assert "test_component" in matches
        assert "production_component" in matches
        assert "other_thing" not in matches

    def test_get_registry_summary(self):
        """Test getting registry summary."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)

        manager.register("string_comp", "test", aliases=["alias1"])
        manager.register("int_comp", 42)

        summary = lister.get_registry_summary()
        assert summary["statistics"]["total_components"] == 2
        assert summary["statistics"]["total_aliases"] == 1
        assert "str" in summary["type_distribution"]
        assert "int" in summary["type_distribution"]

    def test_validate_registry_integrity(self):
        """Test registry integrity validation."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)

        # Create normal registration
        manager.register("good_component", Mock())

        # Manually create integrity issues for testing
        manager._aliases["orphaned_alias"] = "nonexistent_component"
        manager._metadata["orphaned_metadata"] = {"version": "1.0"}
        manager._registry["none_component"] = None

        issues = lister.validate_registry_integrity()

        assert any("Orphaned alias" in issue for issue in issues)
        assert any("Orphaned metadata" in issue for issue in issues)
        assert any("is None" in issue for issue in issues)


class TestComponentCollaboration:
    """Test collaboration between registration components."""

    def test_manager_validator_integration(self):
        """Test that RegistrationManager uses validator properly."""
        validator = RegistrationValidator(reserved_names={"reserved"})
        manager = RegistrationManager(validator)

        with pytest.raises(ValueError, match="Registration validation failed"):
            manager.register("reserved", Mock())

    def test_manager_lister_integration(self):
        """Test that RegistryLister works with RegistrationManager."""
        manager = RegistrationManager()
        lister = RegistryLister(manager)

        # Register some components
        manager.register("comp1", "value1", aliases=["alias1"])
        manager.register("comp2", "value2")

        # Test lister functionality
        components = lister.list_components()
        assert len(components) == 2

        info = lister.get_component_info("alias1")
        assert info is not None
        assert info["name"] == "comp1"
        assert info["component"] == "value1"

        summary = lister.get_registry_summary()
        assert summary["statistics"]["total_components"] == 2
        assert summary["statistics"]["total_aliases"] == 1

    def test_full_workflow(self):
        """Test a complete registration workflow."""
        # Create validator with custom rules
        validator = RegistrationValidator(
            reserved_names={"system", "admin"},
            naming_pattern=r"^[a-z_]+$",  # Only lowercase and underscore
            max_alias_count=3,
        )

        # Create manager with validator
        manager = RegistrationManager(validator)

        # Create lister
        lister = RegistryLister(manager)

        # Register valid components
        manager.register(
            "valid_component",
            Mock(),
            aliases=["alias_one", "alias_two"],
            metadata={"version": "1.0", "description": "Test component"},
        )

        # Try to register invalid component (should fail)
        with pytest.raises(ValueError):
            manager.register("Invalid-Name", Mock())  # Uppercase and hyphen not allowed

        # Check that valid registration worked
        assert manager.is_registered("valid_component")
        assert manager.is_registered("alias_one")

        # Use lister to inspect registry
        components = lister.list_components(include_aliases=True)
        assert "valid_component" in components
        assert "alias_one" in components

        info = lister.get_component_info("valid_component")
        assert info is not None
        assert info["metadata"]["version"] == "1.0"

        # Check integrity
        integrity_issues = lister.validate_registry_integrity()
        assert len(integrity_issues) == 0  # Should be clean
