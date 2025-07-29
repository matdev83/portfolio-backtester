"""
Central registry for all API-stable (protected) methods and their signatures.
"""
import inspect
from typing import Callable, Dict, Any, Optional


import os
import json

class MethodSignature:
    def __init__(self, func: Callable, version: str, strict_params: bool, strict_return: bool):
        self.name = func.__qualname__
        self.module = func.__module__
        self.signature = str(inspect.signature(func))
        self.type_hints = getattr(func, '__annotations__', {})
        self.version = version
        self.strict_params = strict_params
        self.strict_return = strict_return

    def as_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'module': self.module,
            'signature': self.signature,
            'type_hints': self.type_hints,
            'version': self.version,
            'strict_params': self.strict_params,
            'strict_return': self.strict_return,
        }

REFERENCE_SIGNATURES_PATH = os.path.join(os.path.dirname(__file__), 'api_stable_signatures.json')

def load_reference_signatures() -> Dict[str, dict]:
    if os.path.exists(REFERENCE_SIGNATURES_PATH):
        with open(REFERENCE_SIGNATURES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_reference_signatures(refs: Dict[str, dict]):
    with open(REFERENCE_SIGNATURES_PATH, 'w', encoding='utf-8') as f:
        json.dump(refs, f, indent=2, sort_keys=True)

# Global registry: { (module, name): MethodSignature }
method_registry: Dict[str, MethodSignature] = {}

def register_method(func: Callable, version: str, strict_params: bool, strict_return: bool):
    key = f"{func.__module__}.{func.__qualname__}"
    method_registry[key] = MethodSignature(func, version, strict_params, strict_return)


def get_registered_methods() -> Dict[str, MethodSignature]:
    return method_registry

def export_registry_json() -> str:
    import json
    return json.dumps({k: v.as_dict() for k, v in method_registry.items()}, indent=2, default=str)

def get_method_signature(key: str) -> Optional[MethodSignature]:
    return method_registry.get(key)
