import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def small_sales():
    """Two stores x all products, full calendar: fast but keeps every seasonal pattern."""
    from retailfc import data

    full = data.generate(seed=5)
    return full[full["store"].isin(["Al Barsha", "Business Bay"])].reset_index(drop=True)
