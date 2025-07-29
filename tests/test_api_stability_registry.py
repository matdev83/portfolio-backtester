# NOTE:
# If this test fails with a message like
#     "No reference signature stored for <method> …"
# it means a new @api_stable method was added or its signature changed.
# Run the helper script to refresh the snapshot:
#
#     .venv\Scripts\python scripts/update_protected_signatures.py
#
# (Warnings printed by that script are harmless.)
# Commit the updated JSON file alongside your change.
"""
Test that all API-stable methods are registered and their signatures are valid.
"""
import pytest
from portfolio_backtester.api_stability import registry

def test_api_stable_registry_not_empty():
    # Import core to ensure decorated methods are registered
    import portfolio_backtester.core  # noqa: F401
    methods = registry.get_registered_methods()
    assert methods, "No API-stable methods registered. Decorators may not be applied."

@pytest.mark.parametrize("key, meta", list(registry.get_registered_methods().items()))
def test_api_stable_method_signature(key, meta):
    # Check that required fields are present and non-empty
    d = meta.as_dict()
    assert d['name'], f"Missing name for {key}"
    assert d['module'], f"Missing module for {key}"
    assert d['signature'], f"Missing signature for {key}"
    assert isinstance(d['type_hints'], dict), f"Type hints not a dict for {key}"
    assert d['version'], f"Missing version for {key}"
    # Optionally: check for strict param/return flags
    assert isinstance(d['strict_params'], bool)
    assert isinstance(d['strict_return'], bool)

def test_export_registry_json():
    json_str = registry.export_registry_json()
    assert json_str.startswith('{') and 'signature' in json_str


def test_api_stable_signature_enforcement():
    """
    Compare current API-stable method signatures to the persisted reference signatures.
    If they differ, fail the test unless API_STABLE_UPDATE_SIGNATURES=1 is set in the environment.
    """
    import os
    import portfolio_backtester.core  # noqa: F401
    refs = registry.load_reference_signatures()
    current = {k: v.as_dict() for k, v in registry.get_registered_methods().items()}
    update_mode = os.environ.get('API_STABLE_UPDATE_SIGNATURES', '0') == '1'

    # If update mode, overwrite the reference signatures
    if update_mode:
        registry.save_reference_signatures(current)
        return

    # Otherwise, compare
    for key, meta in current.items():
        ref = refs.get(key)
        assert ref is not None, f"No reference signature stored for {key}. Run with API_STABLE_UPDATE_SIGNATURES=1 to update."
        assert meta['signature'] == ref['signature'], (
            f"Signature mismatch for {key}:\nCurrent:   {meta['signature']}\nReference: {ref['signature']}\n"
            "If this change is intentional, run with API_STABLE_UPDATE_SIGNATURES=1 to update the reference."
        )
