"""
Tests for Attribute Accessor DIP (Dependency Inversion Principle) implementation.

This test suite validates the attribute accessor interfaces and their concrete
implementations, ensuring proper dependency injection and SOLID compliance.
"""

import pytest
import logging
from unittest.mock import Mock, patch

from portfolio_backtester.interfaces.attribute_accessor_interface import (
    IAttributeAccessor,
    IModuleAttributeAccessor,
    IClassAttributeAccessor,
    IObjectFieldAccessor,
    create_attribute_accessor,
    create_module_attribute_accessor,
    create_class_attribute_accessor,
    create_object_field_accessor,
)
from portfolio_backtester.interfaces.attribute_accessor_implementations import (
    DefaultAttributeAccessor,
    ModuleAttributeAccessor,
    ClassAttributeAccessor,
    ObjectFieldAccessor,
    LoggingLevelAccessor,
    create_logging_level_accessor,
    create_safe_field_accessor,
)


class TestAttributeAccessorInterfaces:
    """Test attribute accessor interface contracts."""

    def test_default_attribute_accessor_interface_compliance(self):
        """Test that DefaultAttributeAccessor implements IAttributeAccessor interface."""
        accessor = DefaultAttributeAccessor()
        assert isinstance(accessor, IAttributeAccessor)

    def test_module_attribute_accessor_interface_compliance(self):
        """Test that ModuleAttributeAccessor implements IModuleAttributeAccessor interface."""
        accessor = ModuleAttributeAccessor()
        assert isinstance(accessor, IModuleAttributeAccessor)

    def test_class_attribute_accessor_interface_compliance(self):
        """Test that ClassAttributeAccessor implements IClassAttributeAccessor interface."""
        accessor = ClassAttributeAccessor()
        assert isinstance(accessor, IClassAttributeAccessor)

    def test_object_field_accessor_interface_compliance(self):
        """Test that ObjectFieldAccessor implements IObjectFieldAccessor interface."""
        accessor = ObjectFieldAccessor()
        assert isinstance(accessor, IObjectFieldAccessor)

    def test_factory_functions_return_correct_types(self):
        """Test that factory functions return correct interface implementations."""
        default_accessor = create_attribute_accessor()
        module_accessor = create_module_attribute_accessor()
        class_accessor = create_class_attribute_accessor()
        field_accessor = create_object_field_accessor()

        assert isinstance(default_accessor, IAttributeAccessor)
        assert isinstance(module_accessor, IModuleAttributeAccessor)
        assert isinstance(class_accessor, IClassAttributeAccessor)
        assert isinstance(field_accessor, IObjectFieldAccessor)


class TestDefaultAttributeAccessor:
    """Test DefaultAttributeAccessor implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.accessor = DefaultAttributeAccessor()

    def test_get_attribute_with_valid_attribute(self):
        """Test getting valid attribute from object."""

        class TestObj:
            test_attr = "test_value"

        obj = TestObj()
        result = self.accessor.get_attribute(obj, "test_attr")
        assert result == "test_value"

    def test_get_attribute_with_default(self):
        """Test getting attribute with default value."""
        obj = object()
        result = self.accessor.get_attribute(obj, "nonexistent", "default")
        assert result == "default"

    def test_get_attribute_raises_error_without_default(self):
        """Test that AttributeError is raised when attribute doesn't exist and no default."""
        obj = object()
        with pytest.raises(AttributeError):
            self.accessor.get_attribute(obj, "nonexistent")


class TestModuleAttributeAccessor:
    """Test ModuleAttributeAccessor implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.accessor = ModuleAttributeAccessor()

    def test_get_module_attribute_valid(self):
        """Test getting valid module attribute."""
        result = self.accessor.get_module_attribute(logging, "INFO")
        assert result == logging.INFO

    def test_get_module_attribute_with_default(self):
        """Test getting module attribute with default."""
        result = self.accessor.get_module_attribute(logging, "NONEXISTENT", "default")
        assert result == "default"

    def test_get_module_attribute_enhanced_error_message(self):
        """Test enhanced error message for missing module attributes."""
        with pytest.raises(AttributeError) as exc_info:
            self.accessor.get_module_attribute(logging, "NONEXISTENT")

        error_msg = str(exc_info.value)
        assert "logging" in error_msg
        assert "NONEXISTENT" in error_msg


class TestClassAttributeAccessor:
    """Test ClassAttributeAccessor implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.accessor = ClassAttributeAccessor()

    def test_get_class_from_module_valid_class(self):
        """Test getting valid class from module."""
        # Create a mock module with a test class
        mock_module = Mock()
        mock_module.__name__ = "test_module"

        class TestClass:
            pass

        setattr(mock_module, "TestClass", TestClass)

        result = self.accessor.get_class_from_module(mock_module, "TestClass")
        assert result == TestClass

    def test_get_class_from_module_not_a_class(self):
        """Test error when retrieved attribute is not a class."""
        mock_module = Mock()
        mock_module.__name__ = "test_module"
        setattr(mock_module, "NotAClass", "string_value")

        with pytest.raises(TypeError) as exc_info:
            self.accessor.get_class_from_module(mock_module, "NotAClass")

        error_msg = str(exc_info.value)
        assert "is not a class" in error_msg
        assert "test_module" in error_msg

    def test_get_class_from_module_attribute_error(self):
        """Test enhanced error message for missing class."""
        mock_module = Mock()
        mock_module.__name__ = "test_module"
        # Configure mock to not have the attribute
        del mock_module.NonexistentClass  # This will make getattr raise AttributeError
        
        with pytest.raises(AttributeError) as exc_info:
            self.accessor.get_class_from_module(mock_module, "NonexistentClass")
        
        error_msg = str(exc_info.value)
        assert "test_module" in error_msg
        assert "NonexistentClass" in error_msg


class TestObjectFieldAccessor:
    """Test ObjectFieldAccessor implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.accessor = ObjectFieldAccessor()

    def test_get_field_value_valid_field(self):
        """Test getting valid field from object."""

        class TestObj:
            field1 = "value1"

        obj = TestObj()
        result = self.accessor.get_field_value(obj, "field1")
        assert result == "value1"

    def test_get_field_value_with_default(self):
        """Test getting field value with default."""
        obj = object()
        result = self.accessor.get_field_value(obj, "nonexistent", "default")
        assert result == "default"

    def test_get_field_value_enhanced_error_message(self):
        """Test enhanced error message for missing field."""
        obj = object()

        with pytest.raises(AttributeError) as exc_info:
            self.accessor.get_field_value(obj, "nonexistent")

        error_msg = str(exc_info.value)
        assert "object" in error_msg
        assert "nonexistent" in error_msg


class TestLoggingLevelAccessor:
    """Test specialized LoggingLevelAccessor implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.accessor = LoggingLevelAccessor()

    def test_get_valid_logging_level(self):
        """Test getting valid logging level."""
        result = self.accessor.get_module_attribute(logging, "DEBUG")
        assert result == logging.DEBUG

    def test_get_logging_level_with_default(self):
        """Test getting logging level with default fallback."""
        result = self.accessor.get_module_attribute(logging, "INVALID_LEVEL", logging.WARNING)
        assert result == logging.WARNING

    def test_get_logging_level_factory_function(self):
        """Test factory function for logging level accessor."""
        accessor = create_logging_level_accessor()
        assert isinstance(accessor, LoggingLevelAccessor)


class TestSafeFieldAccessor:
    """Test safe field accessor for analysis operations."""

    def setup_method(self):
        """Set up test environment."""
        self.accessor = create_safe_field_accessor()

    def test_safe_field_accessor_returns_default_for_missing(self):
        """Test that safe field accessor always returns a safe default."""
        obj = object()
        result = self.accessor.get_field_value(obj, "nonexistent")
        assert result == ""  # Safe default

    def test_safe_field_accessor_with_custom_default(self):
        """Test safe field accessor with custom default."""
        obj = object()
        result = self.accessor.get_field_value(obj, "nonexistent", "custom_default")
        assert result == "custom_default"

    def test_safe_field_accessor_returns_actual_value(self):
        """Test that safe field accessor returns actual value when field exists."""

        class TestObj:
            existing_field = "actual_value"

        obj = TestObj()
        result = self.accessor.get_field_value(obj, "existing_field")
        assert result == "actual_value"


class TestDependencyInjectionIntegration:
    """Test integration of DIP pattern with timing components."""

    def test_timing_logger_accepts_module_accessor_injection(self):
        """Test that TimingLogger accepts dependency injection."""
        from portfolio_backtester.timing.timing_logger import TimingLogger

        mock_accessor = Mock(spec=IModuleAttributeAccessor)
        mock_accessor.get_module_attribute.return_value = logging.INFO

        # Should not raise exception
        timing_logger = TimingLogger(name="test", module_attribute_accessor=mock_accessor)

        # Verify logger was created and dependency was used
        assert timing_logger.logger is not None
        mock_accessor.get_module_attribute.assert_called_once_with(logging, "INFO", logging.INFO)

    def test_log_analyzer_accepts_field_accessor_injection(self):
        """Test that LogAnalyzer accepts dependency injection."""
        from portfolio_backtester.timing.log_analyzer import LogAnalyzer
        
        mock_accessor = Mock(spec=IObjectFieldAccessor)
        
        # Should not raise exception
        analyzer = LogAnalyzer(field_accessor=mock_accessor)
        
        # Verify dependency was injected
        assert analyzer._field_accessor == mock_accessor

    def test_custom_timing_registry_accepts_class_accessor_injection(self):
        """Test that CustomTimingRegistry accepts dependency injection."""
        from portfolio_backtester.timing.custom_timing_registry import TimingControllerFactory
        from portfolio_backtester.timing.timing_controller import TimingController
        
        mock_accessor = Mock(spec=IClassAttributeAccessor)
        
        # Create a proper mock class that inherits from TimingController
        class MockTimingController(TimingController):
            def should_rebalance(self, current_date, strategy_state, backtest_data):
                return False
                
            def get_rebalance_frequency_days(self):
                return 30
                
            def __init__(self):
                pass  # Skip parent init for test
        
        mock_accessor.get_class_from_module.return_value = MockTimingController
        
        # Mock the import_module to avoid actual import
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_import.return_value = mock_module
            
            # Test static method with dependency injection
            import_result = TimingControllerFactory._import_class(
                "test.module.TestClass", class_accessor=mock_accessor
            )
            
            # Verify result and that injected dependency was used
            assert import_result == MockTimingController
            mock_accessor.get_class_from_module.assert_called_once()


class TestBackwardCompatibility:
    """Test that DIP implementation maintains backward compatibility."""

    def test_timing_logger_works_without_injection(self):
        """Test that TimingLogger works without explicit dependency injection."""
        from portfolio_backtester.timing.timing_logger import TimingLogger

        # Should work with default factory
        logger = TimingLogger(name="test")
        assert logger.logger is not None

    def test_log_analyzer_works_without_injection(self):
        """Test that LogAnalyzer works without explicit dependency injection."""
        from portfolio_backtester.timing.log_analyzer import LogAnalyzer
        
        # Should work with default factory
        analyzer = LogAnalyzer()
        assert analyzer._field_accessor is not None

    def test_custom_timing_registry_works_without_injection(self):
        """Test that CustomTimingRegistry works without explicit dependency injection."""
        from portfolio_backtester.timing.custom_timing_registry import TimingControllerFactory
        from portfolio_backtester.timing.timing_controller import TimingController

        # Mock the import to avoid actual import
        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.__name__ = "test_module"
            mock_import.return_value = mock_module

            # Create proper TimingController subclass
            class MockTimingController(TimingController):
                def should_rebalance(self, current_date, strategy_state, backtest_data):
                    return False
                    
                def get_rebalance_frequency_days(self):
                    return 30
                    
                def __init__(self):
                    pass  # Skip parent init for test

            setattr(mock_module, "MockController", MockTimingController)

            # Should work without explicit dependency injection
            result = TimingControllerFactory._import_class("test.module.MockController")

            # Should use default factory internally and return the class
            assert result == MockTimingController
