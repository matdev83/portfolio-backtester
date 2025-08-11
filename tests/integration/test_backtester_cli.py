import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to the Python path to allow importing the backtester
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
PLOTS_DIR = PROJECT_ROOT / "plots"


def run_cli_backtest(scenario_name: str, timeout_sec: int = 120) -> subprocess.CompletedProcess:
    """Helper to run the backtester CLI in a separate process."""
    command = [
        sys.executable,
        "-m",
        "portfolio_backtester.backtester",
        "--mode",
        "backtest",
        "--scenario-name",
        scenario_name,
        "--log-level",
        "INFO",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            cwd=PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"CLI process timed out after {timeout_sec} seconds: {e}")
    except Exception as e:
        pytest.fail(f"CLI subprocess execution failed: {e}")
    return result


def find_latest_artifact_dir(scenario_name: str) -> Path | None:
    """Finds the most recent report/plot directory for a given scenario."""
    if not REPORTS_DIR.exists():
        return None
    
    # Sanitize scenario name for directory matching
    sanitized_name = scenario_name.replace(" ", "_")
    
    potential_dirs = [p for p in REPORTS_DIR.iterdir() if p.is_dir() and sanitized_name in p.name]
    if not potential_dirs:
        return None
        
    latest_dir = max(potential_dirs, key=lambda p: p.stat().st_mtime)
    return latest_dir


def find_plot_file(artifact_dir: Path, scenario_name: str) -> Path | None:
    """Finds the performance summary plot file within the artifact directory."""
    sanitized_name = scenario_name.replace(" ", "_")
    
    # Check in the main plots directory first
    if PLOTS_DIR.exists():
        potential_plots = list(PLOTS_DIR.glob(f"performance_summary_{sanitized_name}*.png"))
        if potential_plots:
            return max(potential_plots, key=lambda p: p.stat().st_mtime)

    # Fallback to checking inside the report directory's equity curve image
    if artifact_dir.exists():
        equity_curve_path = artifact_dir / "equity_curve.png"
        if equity_curve_path.exists():
            return equity_curve_path
            
    return None


@pytest.fixture(scope="module")
def cleanup_files():
    """Fixture to clean up generated files after tests."""
    generated_dirs = []
    generated_plots = []
    
    def _add_artifacts(dir_path: Path | None, plot_path: Path | None):
        if dir_path and dir_path.exists():
            generated_dirs.append(dir_path)
        if plot_path and plot_path.exists() and plot_path.parent == PLOTS_DIR:
             generated_plots.append(plot_path)

    yield _add_artifacts

    # Teardown: remove all collected artifacts
    for d in generated_dirs:
        try:
            shutil.rmtree(d)
        except OSError as e:
            print(f"Warning: Failed to remove directory {d}: {e}")
    for p in generated_plots:
        try:
            p.unlink()
        except OSError as e:
            print(f"Warning: Failed to remove plot file {p}: {e}")


def test_simple_backtest_cli_execution(cleanup_files):
    """
    Tests a simple end-to-end backtest scenario via the CLI.
    Verifies that the process runs successfully and creates the expected output artifacts.
    """
    scenario_name = "dummy_signal_default"
    
    # --- Act ---
    result = run_cli_backtest(scenario_name)

    # --- Assert ---
    # 1. Check for successful execution
    assert result.returncode == 0, f"CLI process failed with exit code {result.returncode}.\\nSTDERR:\\n{result.stderr}"

    # 2. Find the output artifacts
    artifact_dir = find_latest_artifact_dir(scenario_name)
    assert artifact_dir is not None, f"Could not find artifact directory for scenario '{scenario_name}' in {REPORTS_DIR}"
    assert artifact_dir.exists(), f"Artifact directory {artifact_dir} does not exist."

    plot_file = find_plot_file(artifact_dir, scenario_name)
    assert plot_file is not None, f"Could not find plot file for scenario '{scenario_name}'"
    assert plot_file.exists(), f"Plot file {plot_file} does not exist."

    # 3. Register artifacts for cleanup
    cleanup_files(artifact_dir, plot_file)
