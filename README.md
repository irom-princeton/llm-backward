# Thinking Forward and Backward: Effective Backward Planning with Large Language Models

[Paper](https://arxiv.org/abs/2411.01790)

Allen Z. Ren, Brian Ichter, Anirudha Majumdar

## Setup
Install dependencies (tested with Python 3.10):
```console
pip install -e .  # setup.py
```

Set up OpenAI API key and logging directory
```console
export BACKWARD_OPENAI_KEY=<your_key>
export BACKWARD_LOG_DIR=<your_preferred_logging_directory>
```

## Usage

### Graph search

Run LLM (with CLI arguments) and then evaluate results
```console
python graph/run.py # see file for CLI arguments
python graph/check_results.py --load_path <logged_pickle_path>
```

### Array transformation (PCFG)

Run LLM (with CLI arguments) and then evaluate results
```console
python pcfg/run.py # see file for CLI arguments
python pcfg/check_results.py --load_path <logged_pickle_path>
```

## License
This repository is released under the MIT license. See [LICENSE](LICENSE).