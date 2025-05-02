#!/bin/sh
set -e

# Define constants
# The local path to the injection file IF the user mounted it, e.g.
# docker run --name mateo v "my_custom_injection.html:/injection/injection_file_content:ro" mateo-demo-runtime
INJECTION_FILE_PATH="/injection/injection_file_content"
APP_HOME="/home/mateo_user/mateo-demo"
VENV_PATH="$APP_HOME/.venv"

# --- Activate Virtual Environment ---
# Check if venv exists before trying to activate
if [ -f "$VENV_PATH/bin/activate" ]; then
    . "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH. Container not correctly built. Cannot proceed."
    exit 1
fi

# --- Injection File Handling ---
if [ -f "$INJECTION_FILE_PATH" ]; then
    echo "Processing injection file found at $INJECTION_FILE_PATH..."
    # Ensure the script exists before trying to run it
    if [ -f "$APP_HOME/scripts/patch_index_html.py" ]; then
        python "$APP_HOME/scripts/patch_index_html.py" --input_file "$INJECTION_FILE_PATH" --restore
        echo "Injection file processed."
    else
        echo "Warning: Injection file found, but patch script ($APP_HOME/scripts/patch_index_html.py) not found. Skipping processing."
    fi
else
    echo "No injection file found at $INJECTION_FILE_PATH. Skipping processing."
fi

# --- Optional Model Preloading ---
# Check the PRELOAD_METRICS environment variable (case-sensitive "true")
if [ "$PRELOAD_METRICS" = "true" ]; then
    echo "PRELOAD_METRICS is true. Starting model preloading..."
    python "$APP_HOME/scripts/patch_index_html.py" --config_file "$APP_HOME/configs/precache-demo.json" --precache_translation
else
    echo "PRELOAD_METRICS is not 'true'. Skipping model preloading."
    echo "Models will be downloaded by the application on first use."
fi

# --- Start Streamlit ---
# Set server address
if [ "$SERVER" = "localhost" ]; then
    SERVER_ADDRESS="0.0.0.0"
else
    SERVER_ADDRESS="$SERVER"
fi

# Set streamlit arguments
STREAMLIT_ARGS="--server.port=$PORT --server.address=$SERVER_ADDRESS"
if [ -n "$BASE" ]; then
    STREAMLIT_ARGS="$STREAMLIT_ARGS --server.baseUrlPath $BASE"
fi
if [ "$DEMO_MODE" = "true" ]; then
    STREAMLIT_ARGS="$STREAMLIT_ARGS --server.maxUploadSize 1"
fi
if [ "$ENABLE_XSRF_PROTECTION" = "false" ]; then
    STREAMLIT_ARGS="$STREAMLIT_ARGS --server.enableXsrfProtection=false"
fi

# Set application arguments (passed after --)
APP_ARGS=""
if [ "$DEMO_MODE" = "true" ]; then
    APP_ARGS="$APP_ARGS --demo_mode"
fi
if [ "$USE_CUDA" = "true" ]; then
    APP_ARGS="$APP_ARGS --use_cuda"
fi

echo "Starting Streamlit application..."
exec streamlit run 01_🎈_MATEO.py $STREAMLIT_ARGS -- $APP_ARGS