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

### Usage

#### Bootstrap resampling

See "Statistical Significance Tests for Machine Translation Evaluation" by P. Koehn for more information.

The p-statistic in bootstrap resampling does the following (baseline: bl; system: sys):
- calculate the "real" difference between bl and sys on the full corpus
- calculate scores for all n partitions (e.g. 1000) for the bl and sys. Partitions are drawn from the same set with
replacement. That means that if our dataset contains 300 samples, we create 1000 mini test sets of 300 samples that
are randomly chosen from our initial dataset of 300, but where a sample can occur multiple times. For motivation and
empirical evidence, see the aforementioned publication by Koehn
- calculate the absolute diff between the arrays of bl and system scores
- subtract the mean from this array of absolute diffs. Now it indicates for each partition how "extreme" it is (how
different bl and sys are for this partition), compared to "the average partition"  
- find the number of cases where the absolute difference is larger ("more extreme") than the "real difference"
- divide this no. extreme cases by total no. cases (i.e. n partitions)

What we actually calculated is the probability that for a random subset (with replacement),
bl and sys differ more extremely than their real difference.

If this p value is high, then that means that extreme values (higher than full-corpus diff) are likely to occur.
In turn that also means that we can be _less certain_ that bl and sys _really_ differ significantly.

However, if the p value is low, then that means it is unlikely that for a random set, bl and sys differ
more extremely than for the full corpus (so partition scores are close to full-corpus scores).
That means that we can more certain that bl and sys really differ significantly.

The 95% confidence interval that we can retrieve can be explained as "with a probability of 95%, the real mean
value of this metric for the full population that this dataset comes from, lies between [mean-CI; mean+CI]".
In other words, it tells you how close the calculated metric scores are for all different partitions.

## Acknowledgements

This project was kickstarted by a Sponsorship project from the
[European Association for Machine Translation](https://eamt.org/), and
a substantial follow-up grant by the support of [CLARIN.eu](https://www.clarin.eu/).
