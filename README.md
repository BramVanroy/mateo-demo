# MAchine Translation Evaluation Online

**BETA VERSION**. This version is under active development and will change considerably in the coming months.

We present MAchine Translation Evaluation Online (MATEO), a project that aims to facilitate machine translation (MT)
evaluation by means of an easy-to-use interface that can evaluate given machine translations with a battery of
automatic metrics. It caters to both experienced and novice users who are working with MT, such as MT system builders,
and also researchers from Social Sciences and Humanities, and teachers and students of (machine) translation. The
project is open source and will be hosted at CLARIN.eu infrastructure.

## Running

### Manual

1. Clone the repository
2. Install with `pip install .` (only installing the requirements.txt file is not enough!)
3. `cd src/mateo_st`
4. `streamlit run 01_🎈_MATEO.py`

### Docker

The Dockerfile exposes these environment variables

- PORT: server port to expose and to run the streamlit server on (default: 5004)
- SERVER: server address to run on (default: 'localhost')
- BASE: base path (default: '')
- NO_CUDA: set to `true` to disable CUDA for all operations (default: '')

These will be used in the streamlit command:

```shell
streamlit run 01_🎈_MATEO.py --server.port $PORT --browser.serverAddress $SERVER --server.baseUrlPath $BASE;
```

To build and run the repository on port 5034 with CUDA disabled:

```shell
docker build -t mateo . 
docker run --rm -d --name mateo-demo -p 5034:5034 --env PORT=5054 --env NO_CUDA=true  mateo
```

## Acknowledgements

This project was kickstarted by a Sponsorship project from the
[European Association for Machine Translation](https://eamt.org/), and
a substantial follow-up grant by the support of [CLARIN.eu](https://www.clarin.eu/).
