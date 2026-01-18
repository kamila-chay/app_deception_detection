# HOW TO RUN

## Environment setup

It is recommended to create a fresh Python virtual environment and install all required dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Running the experiments

The experiments can be run either using SLURM (recommended for cluster execution) or by directly executing the Python scripts.


**Option A**: Running with SLURM

Submit the provided shell scripts to the SLURM workload manager:


`sbatch <root>/train/shell_scripts/train_setupA.sh`
etc.

Evaluation can be submitted as:

`sbatch <root>/eval/shell_scripts/eval_setupA_test_set.sh`
etc.

Ensure that the SLURM scripts specify the correct paths, GPU resources, and activate the appropriate Python environment.


**Option B**: Running without SLURM

The Python scripts can also be run directly without SLURM:

`python -m <root>.train.train_setupA`
etc.

Evaluation can be run as:

`python -m <root>.eval.eval_setupA_test_set`
etc.

A CUDA-capable GPU is required. Make sure the correct virtual environment is activated before running the scripts.


## Reproducibility

All experiments use a fixed random seed to ensure reproducibility.
Paths to datasets, checkpoints, and output directories can be adjusted directly in the corresponding scripts.
