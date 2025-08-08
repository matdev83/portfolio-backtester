"""
Tests for study utility functions.

This module tests the study name generation and cleanup utilities
to ensure unique study names and proper cleanup.
"""

import time

from portfolio_backtester.optimization.study_utils import (
    StudyNameGenerator,
    cleanup_test_studies,
    ensure_study_cleanup,
)


class TestStudyNameGenerator:
    """Test study name generation utilities."""

    def test_generate_unique_name_with_timestamp(self):
        """Test unique name generation with timestamp."""
        base_name = "test_study"

        # Generate two names in quick succession
        name1 = StudyNameGenerator.generate_unique_name(base_name, use_timestamp=True)
        time.sleep(0.001)  # Small delay to ensure different timestamps
        name2 = StudyNameGenerator.generate_unique_name(base_name, use_timestamp=True)

        # Both should start with base name
        assert name1.startswith(base_name)
        assert name2.startswith(base_name)

        # Both should be different
        assert name1 != name2

        # Both should contain timestamp
        assert "_" in name1
        assert "_" in name2

    def test_generate_unique_name_with_uuid(self):
        """Test unique name generation with UUID."""
        base_name = "test_study"

        # Generate two names
        name1 = StudyNameGenerator.generate_unique_name(
            base_name, use_timestamp=False, use_uuid=True
        )
        name2 = StudyNameGenerator.generate_unique_name(
            base_name, use_timestamp=False, use_uuid=True
        )

        # Both should start with base name
        assert name1.startswith(base_name)
        assert name2.startswith(base_name)

        # Both should be different
        assert name1 != name2

        # Both should contain UUID (8 characters after underscore)
        assert len(name1.split("_")[-1]) == 8
        assert len(name2.split("_")[-1]) == 8

    def test_generate_unique_name_without_uniqueness(self):
        """Test name generation without uniqueness."""
        base_name = "test_study"

        name = StudyNameGenerator.generate_unique_name(
            base_name, use_timestamp=False, use_uuid=False
        )

        assert name == base_name

    def test_generate_test_study_name(self):
        """Test test-specific study name generation."""
        test_name = "my_test"
        method_name = "test_method"

        # Without method name
        name1 = StudyNameGenerator.generate_test_study_name(test_name)
        assert name1.startswith(f"test_{test_name}_")
        assert len(name1.split("_")[-1]) == 8  # UUID part

        # With method name
        name2 = StudyNameGenerator.generate_test_study_name(test_name, method_name)
        assert name2.startswith(f"test_{test_name}_{method_name}_")
        assert len(name2.split("_")[-1]) == 8  # UUID part

        # Should be different
        assert name1 != name2

    def test_multiple_test_study_names_are_unique(self):
        """Test that multiple test study names are unique."""
        names = []
        for i in range(10):
            name = StudyNameGenerator.generate_test_study_name("test", f"method_{i}")
            names.append(name)

        # All names should be unique
        assert len(names) == len(set(names))


class TestStudyCleanup:
    """Test study cleanup utilities."""

    def test_cleanup_functions_exist(self):
        """Test that cleanup functions exist and can be imported."""
        # This is a basic test to ensure the functions exist
        assert callable(cleanup_test_studies)
        assert callable(ensure_study_cleanup)

        # Test that they handle missing optuna gracefully
        count = cleanup_test_studies("sqlite:///:memory:", "test_")
        assert isinstance(count, int)

        # Test ensure_study_cleanup doesn't crash
        ensure_study_cleanup("sqlite:///:memory:", "nonexistent_study")
