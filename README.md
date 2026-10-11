# Energy-efficient UAV route planning in wind fields using Q-learning

This repository contains the Python code used for the computational experiments in the study of energy-efficient UAV route planning in stationary wind fields under local wind information.

The routing policy is trained with tabular Q-learning. The state includes the UAV position and quantized local wind components. The trained policy is evaluated on previously unseen wind-field realizations and is compared with a Dijkstra reference solution and a local greedy routing algorithm.

## Repository contents

### `main_experiment.py`

Main computational experiment used for the final results.

The script:

- generates seven wind scenarios: nearly uniform, two vortex directions, converging and diverging radial fields, and horizontal and vertical shear fields;
- trains an independent Q-table for each scenario;
- uses three independent training seeds: `0`, `1`, and `2`;
- uses 250,000 training episodes per training run;
- uses the final three-bin wind quantization with thresholds `[-2, 2]` m/s;
- evaluates each trained policy on 50 independent test realizations with seeds `2000...2049`;
- computes Dijkstra and local greedy baselines on the same test realizations;
- saves detailed CSV files, summary files, Q-tables, and figures.

### `quantization_bins_experiment.py`

Preliminary experiment used to select the number of wind-quantization bins.

Three symmetric quantization schemes are compared:

- 3 bins with thresholds `[-1, 1]` m/s;
- 5 bins with thresholds `[-3, -1, 1, 3]` m/s;
- 7 bins with thresholds `[-5, -3, -1, 1, 3, 5]` m/s.

The experiment uses all seven wind scenarios, training seeds `0`, `1`, and `2`, 250,000 training episodes per run, and 50 independent evaluation realizations beginning with seed `1000`.

### `reviewer_experiments.py`

Additional experiments performed during revision of the paper.

Two modes are available through the `EXPERIMENT` variable at the beginning of the file.

`EXPERIMENT = "quantization"` compares three threshold configurations while keeping the number of wind bins fixed at three:

- `[-1, 1]` m/s;
- `[-2, 2]` m/s;
- `[-3, 3]` m/s.

`EXPERIMENT = "reward_sensitivity"` performs a one-at-a-time sensitivity analysis for five reward-function parameters:

- `alpha_d`;
- `beta_L`;
- `delta_dnext`;
- `goal_bonus`;
- `fail_penalty`.

Each parameter is varied by -20% and +20%, using multipliers `0.8` and `1.2`. The sensitivity study uses one representative scenario from each wind-field family: `uniform`, `cyclone_left`, `radial_in`, and `shear_horizontal`. All reported additional experiments use training seeds `0`, `1`, and `2`.

## Requirements

The code is intended for Python 3.8.

Tested dependency versions used for the reported calculations:

- NumPy 1.24.4
- Matplotlib 3.7.5

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

A separate scientific Python stack is not required. The remaining imports are from the Python standard library.

## Recommended environment setup

On Windows, if Python 3.8 is installed:

```bash
py -3.8 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick functional check

Each script contains a `QUICK_TEST` switch near the beginning of the file. For a short smoke test, temporarily set:

```python
QUICK_TEST = True
```

The quick mode uses a strongly reduced number of episodes and evaluation realizations and is intended only to verify that the program runs correctly. Its numerical results must not be interpreted as the results reported in the paper.

After the check, restore:

```python
QUICK_TEST = False
```

before performing the full experiments.

## Running the experiments

### Final main experiment

```bash
python main_experiment.py
```

Results are written to the `main_experiment_results` directory.

### Preliminary 3/5/7-bin comparison

```bash
python quantization_bins_experiment.py
```

The summary is additionally saved as `quantization_bins_results.csv`.

### Three-bin threshold comparison

In `reviewer_experiments.py`, set:

```python
EXPERIMENT = "quantization"
```

and run:

```bash
python reviewer_experiments.py
```

The summary is saved as `review_quantization_thresholds_results.csv`.

### Reward-function sensitivity analysis

In `reviewer_experiments.py`, set:

```python
EXPERIMENT = "reward_sensitivity"
```

and run:

```bash
python reviewer_experiments.py
```

The summary is saved as `review_reward_sensitivity_results.csv`.

## Reproducibility

The scripts use explicit training and evaluation seeds. Separate random-number streams are used for wind-field generation, starting positions, and action selection during training. The main experiment also stores run configuration information together with the results.

Full experiments are computationally intensive. The `QUICK_TEST` mode is provided only for checking the installation and program execution.

## Citation

If you use this code, please cite the corresponding paper:

*Energy-efficient UAV route planning in wind fields using reinforcement learning.*

The complete bibliographic reference and DOI can be added after publication.
