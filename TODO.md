# TODO MATEO

- Add separate "visualization" page for CharCUT?
- Add more options to argparse
- Add "CONTRIBUTING.MD" that explains how to add a metric
- Add significance testing:
  - for sacrebleu metrics, use sacrebleu
  - for other sentence-level metrics, use https://github.com/Unbabel/COMET/blob/db918c6149c771509adcb427e1cf1c6ca94fd926/comet/cli/compare.py#L460
- Add the option to specify cache life expectancy via envvar?
- Add signatures to output
- Remove index from latex and underline "Best" systen (but not that this depends on `higher_better`)
- Add shell script that automatically downloads the most important models/metrics. This can optionally be run before starting
streamlit for the first time to make sure that the basics models are at least downloaded
- Add some kind of pre-batching option so that we can show better progress in the progress bar
- use Ray? https://towardsdatascience.com/parallel-inference-of-huggingface-transformers-on-cpus-4487c28abe23
