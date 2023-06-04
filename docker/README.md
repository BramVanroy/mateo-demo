# Docker files

## hf-spaces

There seems to be a permission/security issue on Hugging Face spaces when using Streamlit inside a Docker container.
The way around this is setting `--server.enableXsrfProtection false`, which is done for you in `hf-spaces/Dockerfile`.