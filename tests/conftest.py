import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def streamlit_server(request):
    # Start Streamlit in a separate process
    root = Path(__file__).parents[1].resolve()
    entrypoint = root / "src" / "mateo_st" / "01_🎈_MATEO.py"
    process = subprocess.Popen(
        [
            "streamlit",
            "run",
            str(entrypoint),
            "--server.port",
            "8505",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--",
            "--no_cuda",
        ]
    )

    # Give the server a bit of time to ensure it's up and running
    time.sleep(5)

    def stop_server():
        # Terminate the Streamlit process when tests are done
        process.terminate()

    request.addfinalizer(stop_server)


@pytest.fixture
def test_data_dir() -> Path:
    test_data_d = Path(__file__).parent.joinpath("test_data")
    assert test_data_d.exists()
    assert test_data_d.is_dir()
    return test_data_d
