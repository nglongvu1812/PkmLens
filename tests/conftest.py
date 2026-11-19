import os
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="cardlens-tests-", dir=Path(__file__).resolve().parents[1] / "data"))
os.environ["DATA_DIR"] = str(TEST_ROOT)
os.environ["DATABASE_URL"] = "sqlite:///" + (TEST_ROOT / "test.db").as_posix()
import pytest
from cardlens_core.db import Base, engine

@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

