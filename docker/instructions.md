# Building and running MATEO with Docker

This project includes a parameterized Dockerfile to build container images for both CPU and GPU environments. An entrypoint script handles runtime configuration, including optional model preloading and HTML injection. All you need is the `Dockerfile` in this directory and follow the configuration steps below to run it.

## Configuration options (both for build time and run time)

The Docker images can be configured using build arguments (during `docker build`) and environment variables (during `docker run`).

**Build-time arguments (`--build-arg`):**

*   **`REPO_BRANCH`**
    *   **Purpose:** specifies the branch, tag, or commit SHA of the `BramVanroy/mateo-demo` repository to clone during the image build.
    *   **Default:** `v1.6.0` (as defined in the Dockerfiles)
    *   **Example:** `docker build --build-arg REPO_BRANCH=main -t mateo-demo:cpu -f Dockerfile.cpu .`

*   **`USE_CUDA`**
    *   **Purpose:** determines if the image should be built with GPU (CUDA 12.1) support. If `true`, the CUDA-enabled version of PyTorch 2.6.0 is installed. If `false`, the CPU-only version is installed.
    *   **Default:** `false`
    *   **Required:** set to `true` when building for GPU usage. Do not forget to enable CUDA during runtime!See `USE_CUDA` in the runtime environment variables.


**Runtime environment variables (`-e` or `--env`):**

*   **`PORT`**
    *   **Purpose:** the port number inside the container on which the Streamlit application will listen.
    *   **Default:** `7860`
    *   **Example:** `docker run -p 8000:8000 -e PORT=8000 ... mateo-demo:cpu` (Remember to map the host port accordingly using `-p`)

*   **`SERVER`**
    *   **Purpose:** the network interface address the Streamlit server should bind to *inside the container*. For accessibility from outside the container, this should generally remain `0.0.0.0` (which the entrypoint script sets automatically if `SERVER` is empty or `localhost`). Setting it to a specific IP might limit accessibility.
    *   **Default:** `localhost` (but effectively becomes `0.0.0.0` via the entrypoint script for external access)
    *   **Example:** generally not needed to override for standard use.

*   **`BASE`**
    *   **Purpose:** sets the base URL path for the Streamlit application (e.g., if running behind a reverse proxy under a subpath like `/mateo`).
    *   **Default:** `""` (empty string, meaning root path `/`)
    *   **Example:** `docker run -e BASE=/mateo ... mateo-demo:cpu` (Access via `http://localhost:7860/mateo`)

*   **`ENABLE_XSRF_PROTECTION`**
    *  **Purpose:** enables support for Cross-Site Request Forgery (XSRF) protection when running Streamlit. Usually kept true but notably on Hugging Face spaces this should be disabled
    *   **Default:** enabled (`true`)
    *   **Example:** `docker run -e ENABLE_XSRF_PROTECTION=false ... mateo-demo:cpu`

*   **`DEMO_MODE`**
    *   **Purpose:** enables/disables demo-specific settings within the Streamlit application (like reduced upload size limits and passing the `--demo_mode` flag to limit the metric options that are available).
    *   **Default:** `false`
    *   **Example:** `docker run -e DEMO_MODE=true ... mateo-demo:cpu`

*   **`HF_HUB_ENABLE_HF_TRANSFER`**
    *   **Purpose:** enables a potentially faster library (`hf_transfer`) for downloading files from the Hugging Face Hub.
    *   **Default:** `1` (enabled)
    *   **Example:** `docker run -e HF_HUB_ENABLE_HF_TRANSFER=0 ... mateo-demo:cpu` (to disable)

*   **`USE_CUDA`**
    *   **Purpose:** informs the application whether a GPU should be used for translation (GPU-accelerated metrics not supported). This is useful if you have built your container with CUDA-support and want to enable it during runtime, or keep it disabled to avoid using the GPU. The entrypoint passes the `--use_cuda` flag to the application if this is `true`.
    *   **Default:** `false` (in `Dockerfile.cpu`), `true` (in `Dockerfile.gpu`)
    *   **Example:** usually not overridden manually, rely on the correct image tag.

*   **`PRELOAD_METRICS`**
    *   **Purpose:** if set to `true`, the `entrypoint.sh` script will attempt to download required models/metrics when the container starts, before Streamlit is started. Otherwise, models are downloaded by the application on first use but this will lead to the first user having to wait while models/metrics are being downloaded.
    *   **Default:** `true`
    *   **Example:** `docker run -e PRELOAD_METRICS=false ... mateo-demo:cpu` (to disable preloading)

**Runtime volume mounts (`-v` or `--volume`):**

*   **Cache directory (`/home/mateo_user/.cache`)**
    *   **Purpose:** mount a persistent volume here to cache downloaded models and metrics across container restarts.
    *   **Example:** `-v mateo-cache:/home/mateo_user/.cache` (using a named volume) or `-v /path/on/host/cache:/home/mateo_user/.cache` (using a host directory).

*   **Injection file (`/injection/injection_file_content`)**
    *   **Purpose:** mount a custom JS file here to be injected into the application's `<head>` at startup.
    *   **Example:** `-v /path/on/host/to/your_injection.html:/injection/injection_file_content:ro` (mount read-only).

## Building the image

*   **CPU Image:**
    ```bash
    # Using default, most recent REPO_BRANCH
    docker build -t mateo-demo:cpu -f Dockerfile .

    # Specifying a different branch/tag
    docker build --build-arg REPO_BRANCH=v1.6 -t mateo-demo:cpu-main -f Dockerfile .
    ```

*   **GPU Image:**
    *(Ensure your system has the necessary NVIDIA drivers and nvidia-docker installed)*
    ```bash
    docker build --build-arg USE_CUDA=true -t mateo-demo:gpu -f Dockerfile .
    ```

## Running the container

Combine `docker run` flags as needed based on the configuration options above.

*   **Basic Run (CPU):**
    ```bash
    docker run -p 7860:7860 --rm --name mateo mateo-demo:cpu
    ```

*   **GPU Run:**
    ```bash
    docker run --gpus all -p 7860:7860 --rm --name mateo -e USE_CUDA=true mateo-demo:gpu
    ```

*   **Example with Docker volume cache, preloading, and injection:**
    ```bash
    # Create volume if it doesn't exist
    docker volume create mateo-cache

    docker run -p 7860:7860 --rm --name mateo \
      -e PRELOAD_METRICS=true \
       -e PORT=7860 \
      -v mateo-cache:/home/mateo_user/.cache \
      -v /path/on/host/to/your_injection.html:/injection/injection_file_content:ro \
      mateo-demo:cpu
    ```

    Note: do not change `/injection/injection_file_content`; this path is used by the entrypoint!

*   **Example changing port and enabling demo mode:**
    ```bash
    docker run -p 8000:8000 --rm --name mateo \
      -e PORT=8000 \
      -e DEMO_MODE=true \
      mateo-demo:cpu
    ```

Access the application in your browser, typically at `http://localhost:7860` (or the host port you mapped and base path you configured).
