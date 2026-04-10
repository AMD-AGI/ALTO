import sys
import types
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
MXFP8_DIR = TESTS_DIR.parent
KERNELS_DIR = MXFP8_DIR.parent
MODELOPTIMIZER_DIR = KERNELS_DIR.parent
REPO_ROOT = MODELOPTIMIZER_DIR.parent


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module

    module.__file__ = str(path / "__init__.py")
    module.__path__ = [str(path)]
    return module


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

modeloptimizer_pkg = _ensure_package("modeloptimizer", MODELOPTIMIZER_DIR)
kernels_pkg = _ensure_package("modeloptimizer.kernels", KERNELS_DIR)
mxfp8_pkg = _ensure_package("modeloptimizer.kernels.mxfp8", MXFP8_DIR)
tests_pkg = _ensure_package("modeloptimizer.kernels.mxfp8.tests", TESTS_DIR)

modeloptimizer_pkg.kernels = kernels_pkg
kernels_pkg.mxfp8 = mxfp8_pkg
mxfp8_pkg.tests = tests_pkg
