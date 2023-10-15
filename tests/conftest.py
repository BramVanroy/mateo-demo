import subprocess
from pathlib import Path

import pytest
import streamlit as st


@pytest.fixture(scope="session", autouse=True)
def streamlit_server(request):
    """Starts a streamlit server and sets `pytest.mateo_st_local_url` that contains local URL"""
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
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    # Read process output to wait for startup and to get the correct local IP address/URL
    for line in iter(process.stdout.readline, ""):
        line = line.strip()
        if line and line.startswith("Network URL:"):
            pytest.mateo_st_local_url = line.split()[-1]
            break

    def stop_server():
        # Terminate the Streamlit process when tests are done
        process.terminate()

    request.addfinalizer(stop_server)


@pytest.fixture(autouse=True)
def reset_st_session_state():
    """Clean up the session state to its default after each test"""
    for key in st.session_state.keys():
        del st.session_state[key]
    yield


@pytest.fixture
def test_data_dir() -> Path:
    test_data_d = Path(__file__).parent.joinpath("test_data")
    assert test_data_d.exists()
    assert test_data_d.is_dir()
    return test_data_d
