# How to run the MC–MD and GB segregation analysis workflow

This document describes how to run the MC–MD workflow and the grain-boundary segregation post-processing script.

## 1. MC–MD workflow script

Script:

`script/mc_md_workflow_distributed_mace.py`

Correct path:

`scripts/mc_md_workflow_distributed_mace.py`

## Required input for MC–MD

The MC–MD script expects the starting grain-boundary structure to be named:

`initial.xyz`

Place `initial.xyz` in the same folder where the MC–MD script is run.

## Temperature selection

The target temperature is read automatically from the name of the run folder.

Accepted folder-name examples include:

* `300`
* `300K`
* `T300`
* `temp300`
* `2000`
* `2000K`
* `T2000`
* `temp2000`

For example, to run a 300 K simulation, run the script inside a folder named:

`300K`

## Interatomic potential

The manuscript simulations used the public pretrained MACE-OMAT-0 small model without author training, retraining, or fine-tuning.

The MC–MD script expects the model file to be available as:

`small.model`

Users may either rename the downloaded public MACE model file to `small.model` or edit the `model_paths` variable in the script.

## Calculator note

The uploaded MC–MD script corresponds to the distributed-GPU version used on the HPC system. It uses a distributed MACE calculator for multi-GPU execution.

For non-distributed calculations, users may replace the distributed calculator section with a standard ASE-compatible MACE calculator or another ASE-compatible machine-learning interatomic potential.

## Main MC–MD settings

Important settings are defined near the top of the script:

* `swap_steps`
* `initial_relax_steps`
* `short_md_steps`
* `timestep_fs`
* `non_swappable_species`
* `snapshot_interval`
* `snapshot_window`

In the manuscript simulations, carbon was treated as non-swappable and metal atoms were allowed to swap.

## MC–MD output files

The MC–MD script writes:

* `all_swaps.csv`
* `accepted_swaps.csv`
* `md_summary.csv`
* `restart_state.json`
* `best_structure.xyz`
* `run_info.txt`
* `accepted_snapshots/*.xyz`

If `restart_state.json` and `best_structure.xyz` are present, the script resumes from the previous MC step.

## Run command for MC–MD

Example:

```bash
python scripts/mc_md_workflow_distributed_mace.py
```

or, if running from inside a temperature folder:

```bash
python ../scripts/mc_md_workflow_distributed_mace.py
```

---

# 2. GB segregation post-processing script

Script:

`scripts/gb_segregation_postprocessing.py`

## Purpose

This script analyzes `.xyz` files from MC–MD grain-boundary simulations and calculates distance-resolved metal-sublattice composition profiles as a function of distance from the nearest grain boundary.

The script also identifies the dominant GB-enriched species and calculates segregation/selectivity metrics.

## Required input for post-processing

Place the `.xyz` files to be analyzed in the folder where the post-processing script is run.

The script automatically processes all files ending with:

`.xyz`

in the current folder.

## Geometry assumption

The script assumes that the grain boundaries are located at:

* `x = 0`
* `x = a/2`

where `a` is the simulation cell length along the x-direction.

The distance assigned to each atom is the distance to the nearest GB plane.

## Species analyzed

The script analyzes only the metal sublattice. Carbon is excluded from the metal-composition normalization.

The supported metal species are:

`Cr, Hf, Mo, Nb, Ta, Ti, V, W, Zr`

Therefore, the script can be applied to different high-entropy carbide compositions containing these elements.

## Binning method

By default, the script uses equal-count distance bins:

`EQUAL_COUNT = True`

The GB region used for top-segregant analysis is controlled by:

`GB_MODE = "first_n_bins"`

and:

`FIRST_N_BINS = 2`

Therefore, by default, the first two bins closest to the GB are used as the GB-core region.

## Post-processing output files

For each input `.xyz` file, the script writes:

* distance-binned composition CSV file
* PNG composition plot
* PDF composition plot

The script also writes two summary files:

* `gb_top_segregant_selectivity_summary.csv`
* `gb_all_species_ranked_enrichment.csv`

## Run command for post-processing

From the folder containing the `.xyz` files, run:

```bash
python path/to/scripts/gb_segregation_postprocessing.py
```

Example:

```bash
python ../scripts/gb_segregation_postprocessing.py
```

## Important note

The post-processing script processes all `.xyz` files in the current directory. Therefore, run it in a folder containing only the `.xyz` files intended for analysis.

Large raw trajectory files are not included in this repository. Users should provide their own `.xyz` simulation snapshots or trajectories.
