import os
import shutil
from pathlib import Path


def is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def clear_pycache(project_root: Path) -> int:
    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    deleted_dirs = 0
    for path in project_root.rglob("__pycache__"):
        if path.is_dir() and is_subpath(path, project_root):
            shutil.rmtree(path, ignore_errors=True)
            deleted_dirs += 1
    return deleted_dirs


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    cwd = Path(os.getcwd()).resolve()
    if not is_subpath(cwd, project_root):
        raise RuntimeError(
            f"Refusing to run outside project root. cwd={cwd} project_root={project_root}"
        )

    deleted = clear_pycache(project_root)
    print(f"Removed {deleted} __pycache__ directories under {project_root}")


if __name__ == "__main__":
    main()
