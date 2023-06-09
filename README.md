# MAchine Translation Evaluation Online (MATEO)

<p align="center">
  <a href="https://huggingface.co/spaces/BramVanroy/mateo-demo" target="_blank"><img alt="HF Spaces shield" src="https://img.shields.io/badge/%F0%9F%A4%97-%20HF%20Spaces-orange?style=flat"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0" target="_blank"><img alt="License shield" src="https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat"></a>
  <img alt="Code style black" src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat">
  <img alt="Built with Streamlit" src="https://img.shields.io/static/v1?style=for-the-badge&message=Streamlit&color=FF4B4B&logo=Streamlit&logoColor=FFFFFF&label&style=flat">
</p>


We present MAchine Translation Evaluation Online (MATEO), a project that aims to facilitate machine translation (MT)
evaluation by means of an easy-to-use interface that can evaluate given machine translations with a battery of
automatic metrics. It caters to both experienced and novice users who are working with MT, such as MT system builders,
and also researchers from Social Sciences and Humanities, and teachers and students of (machine) translation.

MATEO can be accessed on [this website](https://lt3.ugent.be/mateo/), currently hosted by LT3 but soon
moving to the CLARIN B center at [Instituut voor de Nederlandse Taal](https://ivdnt.org/). It also available on Hugging
Face [Spaces](https://huggingface.co/spaces/BramVanroy/mateo-demo).

## Self-hosting

### Duplicating a Hugging Face Spaces



### Install locally with Python

You can clone and install the library on your own device (laptop, computer, server). I recommend to run this in a new 
virtual environment. It requires `python >= 3.8`.

Run the following commands:

```shell
git clone https://github.com/BramVanroy/mateo-demo.git
cd mateo-demo
python -m pip install .
cd src/mateo_st
streamlit run 01_🎈_MATEO.py
```

The streamlit server will then start on your own computer. You can access the website via a local address,
[http://localhost:8501](http://localhost:8501) by default. 

Configuration options specific to Streamlit can be found
[here](https://docs.streamlit.io/library/advanced-features/configuration). They are more related to server-side configurations
that you typically do not need when you are running this directly through Python. But you may need them when you are
using Docker, e.g. setting the `--server.port` that streamlit is running on (see [Docker](#running-with-docker)).

A number of command-line arguments are available to change the interface to your needs.

```shell
--no_cuda             whether to disable CUDA for all tasks (default: False)                                                                                                                                      
--transl_no_cuda      whether to disable CUDA for translation only (default: False)                                                                                                                               
--transl_batch_size TRANSL_BATCH_SIZE                                                                                                                                                                             
                    batch size for translating (default: 8)                                                                                                                                                     
--transl_no_quantize  whether to disable CUDA torch quantization of the translation model. Quantization makes the model smaller and faster but may result in lower quality. This option will disable quantization.
                    (default: False)                                                                                                                                                                            
--transl_model_size {distilled-600M,1.3B,distilled-1.3B,3.3B}                                                                                                                                                     
                    translation model size to use (default: distilled-600M)                                                                                                                                     
--transl_num_beams TRANSL_NUM_BEAMS                                                                                                                                                                               
                    number of beams to allow to generate translations with (default: 1)
--transl_max_length TRANSL_MAX_LENGTH
                    maximal length to generate per sentence (default: 128)
--eval_max_sys EVAL_MAX_SYS
                    max. number of systems to compare (default: 4)
--demo_mode           when demo mode is enabled, only a limited range of neural check-points are available. So all metrics are available but not all of the checkpoints. (default: False)
--config CONFIG       an optional JSON config file that contains script arguments. NOTE: options specified in this file will overwrite those given in the command-line. (default: None)
```

These can be passed to the Streamlit launcher by adding a `--` after the streamlit command and streamlit-specific
options, followed by any of the options above.

For instance, if you want to run streamlit specifically on port 1234 and you want to use the 3.3B version of the
NLLB translation model and to set the max. number of systems that can be evaluated simultaneously, you can modify
your command to look like this:

```shell
streamlit run 01_🎈_MATEO.py --server.port 1234 -- --transl_model_size 3.3B --eval_max_sys 42 
```

Note the separating `--` in the middle so that streamlit can distinguish between streamlit's own options and the MATEO
configuration parameters.


### Running with Docker

If you have docker installed, it is very easy to get a MATEO instance running.

The following Dockerfiles are available in the [`docker`](docker) directory. They are a little bit different depending
on the specific needs.

- [`hf-spaces`](docker/hf-spaces/Dockerfile): specific configuration for Hugging Face spaces but without env options
- [`default`](docker/default/Dockerfile): a more intricate Dockerfile that accepts environment variables to be used
that are specific to the server, demo functionality, and CUDA. These Docker environment variables are available.

  - PORT: server port to expose and to run the streamlit server on (default: 7860)
  - SERVER: server address to run on (default: 'localhost')
  - BASE: base path (default: '')
  - NO_CUDA: set to `true` to disable CUDA for all operations (default: '')
  - DEMO_MODE: set to `true` to disable some options for neural metrics and to limit the max. upload size to 1MB 
  per file (default: '')

As an example, to build and run the repository on port 5034 with CUDA disabled, you can run the following commands
from the root of this directory.

```shell
docker build -t mateo docker/default 
docker run --rm -d --name mateo-demo -p 5034:5034 --env PORT=5054 --env NO_CUDA=true mateo
```

As mentioned before, you can modify the Dockerfiles as you wish. Most notably you may want to change the `streamlit`
launcher command itself. Therefore you could use the [streamlit options]([here](https://docs.streamlit.io/library/advanced-features/configuration))
alongside custom options for MATEO specifically, which were mentioned in the [previous section](#install-locally-with-python).


## Acknowledgements

This project was kickstarted by a Sponsorship project from the
[European Association for Machine Translation](https://eamt.org/), and
a substantial follow-up grant by the support of [CLARIN.eu](https://www.clarin.eu/).
