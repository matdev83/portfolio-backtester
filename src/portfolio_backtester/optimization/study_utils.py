"""
Utility functions for Optuna study management.

This module provides utilities for generating unique study names and managing
study lifecycle to prevent conflicts in testing and parallel execution.
"""

import time
import uuid
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StudyNameGenerator:
    """Generator for unique Optuna study names."""

    @staticmethod
    def generate_unique_name(
        base_name: str, use_timestamp: bool = True, use_uuid: bool = False
    ) -> str:
        """
        Generate a unique study name based on a base name.

        Args:
            base_name: Base name for the study
            use_timestamp: Whether to include timestamp for uniqueness
            use_uuid: Whether to include UUID for uniqueness (overrides timestamp)

        Returns:
            Unique study name
        """
        if use_uuid:
            unique_id = uuid.uuid4().hex[:8]
            return f"{base_name}_{unique_id}"
        elif use_timestamp:
            timestamp = int(time.time() * 1000)  # milliseconds for better uniqueness
            return f"{base_name}_{timestamp}"
        else:
            return base_name

    @staticmethod
    def generate_test_study_name(test_name: str, method_name: Optional[str] = None) -> str:
        """
        Generate a unique study name for testing.

        Args:
            test_name: Name of the test class or module
            method_name: Name of the test method (optional)

        Returns:
            Unique test study name
        """
        base_name = f"test_{test_name}"
        if method_name:
            base_name = f"{base_name}_{method_name}"

        # Use UUID for tests to ensure uniqueness even in rapid succession
        return StudyNameGenerator.generate_unique_name(
            base_name, use_timestamp=False, use_uuid=True
        )


def cleanup_test_studies(storage_url: str, study_name_pattern: str = "test_") -> int:
    """
    Clean up test studies from the storage.

    Args:
        storage_url: URL of the Optuna storage
        study_name_pattern: Pattern to match test study names

    Returns:
        Number of studies cleaned up
    """
    try:
        import optuna

        # Get all studies
        storage = optuna.storages.get_storage(storage_url)
        study_summaries = storage.get_all_studies()

        cleaned_count = 0
        for summary in study_summaries:
            if study_name_pattern in summary.study_name:
                try:
                    storage.delete_study(summary._study_id)
                    cleaned_count += 1
                    logger.debug(f"Cleaned up test study: {summary.study_name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up study {summary.study_name}: {e}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} test studies")

        return cleaned_count

    except ImportError:
        logger.warning("Optuna not available for study cleanup")
        return 0
    except Exception as e:
        logger.error(f"Error during study cleanup: {e}")
        return 0


def ensure_study_cleanup(storage_url: str, study_name: str) -> None:
    """
    Ensure a specific study is cleaned up and remove local SQLite file if used.

    Args:
        storage_url: URL of the Optuna storage
        study_name: Name of the study to clean up
    """
    try:
        import optuna
        import os
        from urllib.parse import urlparse

        storage = optuna.storages.get_storage(storage_url)
        study_summaries = storage.get_all_studies()

        for summary in study_summaries:
            if summary.study_name == study_name:
                try:
                    storage.delete_study(summary._study_id)
                    logger.debug(f"Cleaned up study: {study_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to clean up study {study_name}: {e}")

        parsed = urlparse(storage_url)
        if parsed.scheme == "sqlite" and parsed.path:
            db_path = parsed.path
            if (
                db_path.startswith("/")
                and os.name == "nt"
                and len(db_path) > 1
                and db_path[2:3] == ":"
            ):
                db_path = db_path[1:]
            try:
                if os.path.exists(db_path) and db_path != ":memory:":
                    os.remove(db_path)
                    logger.debug(f"Removed SQLite storage file: {db_path}")
            except Exception as e:
                logger.warning(f"Failed to remove SQLite file {db_path}: {e}")

    except ImportError:
        logger.warning("Optuna not available for study cleanup")
    except Exception as e:
        logger.error(f"Error cleaning up study {study_name}: {e}")
