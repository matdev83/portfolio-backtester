import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.strategy_config_cross_validator import StrategyConfigCrossValidator

class TestStrategyConfigCrossValidator:
    @pytest.fixture
    def mock_dirs(self, tmp_path):
        src_strategies = tmp_path / "src" / "portfolio_backtester" / "strategies"
        config_scenarios = tmp_path / "config" / "scenarios"
        src_strategies.mkdir(parents=True)
        config_scenarios.mkdir(parents=True)
        return src_strategies, config_scenarios

    @pytest.fixture
    def validator(self, mock_dirs):
        src_dir, config_dir = mock_dirs
        # Mock strategy resolver to avoid scanning real registry
        with patch("src.portfolio_backtester.strategy_config_cross_validator.StrategyResolverFactory") as mock_factory:
            mock_resolver = MagicMock()
            mock_factory.create.return_value = mock_resolver
            
            # Setup validator
            val = StrategyConfigCrossValidator(str(src_dir), str(config_dir))
            
            # Helper to mock valid strategy names for tests
            val._get_valid_strategy_names = MagicMock(return_value={"ValidStrategy", "valid_strategy"})
            val.strategy_resolver = mock_resolver
            
            return val

    def test_check_stale_config_folders(self, validator, mock_dirs):
        src_dir, config_dir = mock_dirs
        
        # Create a valid config folder structure
        # config/scenarios/portfolio/valid_strategy (matches valid name)
        valid_conf_dir = config_dir / "portfolio" / "valid_strategy"
        valid_conf_dir.mkdir(parents=True)
        
        # Create a STALE config folder structure
        # config/scenarios/portfolio/stale_strategy (no matching strategy)
        stale_conf_dir = config_dir / "portfolio" / "stale_strategy"
        stale_conf_dir.mkdir(parents=True)
        
        # Configure resolver to confirm stale strategy is unknown
        validator.strategy_resolver.resolve_strategy.side_effect = lambda x: None if x == "stale_strategy" else MagicMock()
        
        errors = validator._check_stale_config_folders()
        
        # Should find error for stale_strategy
        assert len(errors) == 1
        assert "Stale config folder detected" in errors[0]
        assert "stale_strategy" in errors[0]

    def test_validate_yaml_strategy_references_valid(self, validator, mock_dirs):
        src_dir, config_dir = mock_dirs
        yaml_file = config_dir / "test.yaml"
        yaml_file.write_text("strategy: ValidStrategy\nstrategy_params: {}", encoding="utf-8")
        
        valid_names = {"ValidStrategy"}
        
        errors = validator._validate_yaml_strategy_references(yaml_file, valid_names)
        assert len(errors) == 0

    def test_validate_yaml_strategy_references_invalid(self, validator, mock_dirs):
        src_dir, config_dir = mock_dirs
        yaml_file = config_dir / "test.yaml"
        yaml_file.write_text("strategy: InvalidStrategy\nstrategy_params: {}", encoding="utf-8")
        
        valid_names = {"ValidStrategy"}
        # Ensure resolver also returns None
        validator.strategy_resolver.resolve_strategy.return_value = None
        
        errors = validator._validate_yaml_strategy_references(yaml_file, valid_names)
        assert len(errors) == 1
        assert "Invalid strategy reference" in errors[0]
        assert "InvalidStrategy" in errors[0]

    def test_check_meta_strategy_allocations(self, validator, mock_dirs):
        src_dir, config_dir = mock_dirs
        yaml_file = config_dir / "meta.yaml"
        
        config_data = {
            "strategy": "MetaStrategy",
            "strategy_params": {
                "allocations": [
                    {"strategy_id": "ValidStrategy", "weight": 0.5},
                    {"strategy_id": "InvalidSubStrategy", "weight": 0.5}
                ]
            }
        }
        
        valid_names = {"ValidStrategy"}
        # Mock resolver behavior
        def resolve_side_effect(name):
            if name == "ValidStrategy": return MagicMock()
            return None
        validator.strategy_resolver.resolve_strategy.side_effect = resolve_side_effect
        
        errors = validator._check_meta_strategy_allocations(yaml_file, config_data, valid_names)
        
        assert len(errors) == 1
        assert "Invalid strategy reference in meta-strategy allocation" in errors[0]
        assert "InvalidSubStrategy" in errors[0]

    def test_check_python_file_strategy_references(self, validator, mock_dirs):
        src_dir, config_dir = mock_dirs
        py_file = src_dir / "my_strategy.py"
        
        # Python code referencing strategies
        code = """
        class MyStrategy:
            def setup(self):
                # Valid reference
                s1 = _resolve_strategy("ValidStrategy")
                # Invalid reference
                s2 = _resolve_strategy("MissingStrategy")
        """
        py_file.write_text(code, encoding="utf-8")
        
        # Override valid names for this test
        validator._get_valid_strategy_names = MagicMock(return_value={"ValidStrategy"})
        validator.strategy_resolver.resolve_strategy.side_effect = lambda x: MagicMock() if x == "ValidStrategy" else None
        
        errors = validator._check_python_file_strategy_references(py_file)
        
        assert len(errors) == 1
        assert "Potential invalid strategy reference" in errors[0]
        assert "MissingStrategy" in errors[0]

