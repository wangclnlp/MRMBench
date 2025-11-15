# Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models

## Introduction
*MRMBench* is a **M**ulti-dimensional **R**eward **M**odel **Bench**mark, including a collection of six probing tasks for different preference dimensions. 
We design it to favor and encourage reward models that better capture preferences across different dimensions. 

## Getting Started


### Installation
You can use anaconda/miniconda to install packages needed for this project.

```bash
pip install -r requirements.txt
```
### Data

See [our huggingface](https://huggingface.co/datasets/ifnoc/MRMBench) for more details.

### Evaluation

For MRMBench-Easy evaluation:
```bash
bash ./scripts/evaluate_MRMBench-Easy.sh
```

For MRMBench-Hard evaluation:
```bash
bash ./scripts/evaluate_MRMBench-Hard.sh
```