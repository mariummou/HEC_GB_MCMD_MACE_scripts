# HEC_GB_MCMD_MACE_scripts
Scripts and instructions for MC–MD grain-boundary segregation analysis in high-entropy carbides using a public pretrained MACE-OMAT model; adaptable to other ASE-compatible MLIPs.

## Scope of this repository

This repository provides scripts and documentation for:
Generation of composition-specific Σ5(210) HEC grain-boundary structures from a provided `initial.xyz` template.
MC–MD atom-swap sampling of high-entropy carbide grain-boundary structures.
Distance-resolved grain-boundary metal-sublattice composition analysis.
Identification of dominant GB-enriched species.
Calculation of GB enrichment and segregation selectivity metrics.
Generation of GB composition-profile plots.

This repository is intended to document and share the computational workflow used in the manuscript. The MC–MD workflow is written around an ASE-compatible calculator. Therefore, the workflow can be adapted to other ASE-compatible machine-learning interatomic potentials (MLIPs), provided that the calculator section in the script is modified appropriately by the user.

## Applicability

The post-processing workflow can be applied to different high-entropy carbide compositions, provided that the input structures follow the same grain-boundary geometry convention used in the script.

In the associated manuscript, the analyzed structures correspond to a Σ5(210) grain boundary with a tilt angle of 53.1°. The post-processing script assumes a periodic bicrystal/polycrystal geometry in which the grain-boundary planes are located at:

x = 0
x = a/2

where a is the simulation cell length along the x-direction. The distance of each atom is calculated with respect to the nearest grain-boundary plane.

The workflow can therefore be applied to other HEC systems if:

the input structures are readable by ASE;
the grain-boundary geometry follows the same convention used in the script;
the GB planes are located at x = 0 and x = a/2;
the structures use the same or compatible Σ5(210)-type GB setup, or the script is modified for a different GB geometry;
the metal species are included in the supported metal list;
the user provides the appropriate input .xyz files.

The currently supported metal species in the post-processing script are:

Cr, Hf, Mo, Nb, Ta, Ti, V, W, Zr

Carbon is excluded from the metal-sublattice composition normalization.

## Repository contents

This repository contains the following main files and folders:

```text
HEC_GB_MCMD_MACE_scripts/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── initial.xyz
├── scripts/
│   ├── make_sigma5_210_gb_from_initial.py
│   ├── mc_md_workflow.py
│   └── gb_segregation_postprocessing.py
└── docs/
    └── how_to_run.md
```

## Scripts
## 1. GB structure/data creation script

`scripts/make_sigma5_210_gb_from_initial.py`

This script generates composition-specific Σ5(210) high-entropy carbide grain-boundary structures from the provided `initial.xyz` template.

The `initial.xyz` file contains the fixed Σ5(210) 53.1° ⟨001⟩ symmetric tilt grain-boundary geometry. The script preserves the atomic coordinates, simulation cell, carbon sublattice, and microscopic GB motif from `initial.xyz`, while randomly assigning the requested metal species on the metal sublattice.

The GB structure follows this convention:

```text
GB type:        Σ5(210) symmetric tilt grain boundary
Misorientation: 53.1°
Tilt axis:      ⟨001⟩
GB plane:       (210)
GB normal:      x direction
Periodic GBs:   x = 0/Lx and x = Lx/2
```

## 2. MC–MD workflow script
scripts/mc_md_workflow.py

This script performs MC atom-swap sampling combined with short MD relaxation using a MACE calculator.

Main functions of the script:

reads the starting structure from initial.xyz;
automatically identifies chemical species from the input structure;
treats carbon as non-swappable by default;
allows metal atoms to swap;
reads the target temperature from the folder name;
performs an initial NPT relaxation;
performs MC atom swaps with a Metropolis acceptance criterion;
performs short MD relaxation after accepted and rejected MC steps;
saves accepted swap information and MD summary files;
saves sparse accepted snapshots to reduce file count;
supports restart using restart_state.json and best_structure.xyz.

The script expects the MACE model file to be available as:

small.model

or the user should edit the model_paths variable in the script.

## 3. GB segregation post-processing script
scripts/gb_segregation_postprocessing.py

This script analyzes .xyz files from MC–MD grain-boundary simulations and calculates distance-resolved metal-sublattice composition profiles as a function of distance from the nearest GB.

Main functions of the script:

reads all .xyz files in the current folder;
calculates the distance of each metal atom from the nearest GB plane;
bins atoms based on distance from the GB;
calculates metal-sublattice composition in each bin;
excludes carbon from metal-composition normalization;
generates CSV files containing distance-binned compositions;
generates PNG and PDF composition-profile plots;
identifies the dominant GB-enriched species;
calculates GB enrichment relative to the bulk region;
calculates a segregation selectivity metric.

By default, the script uses the first two bins closest to the GB as the GB-core region for top-segregant analysis.

## Required Python packages

The main Python packages are listed in requirements.txt.

Typical requirements include:

numpy
pandas
matplotlib
scipy
ase
torch
mace-torch
pyyaml

The MC–MD script used in the manuscript corresponds to the distributed-GPU workflow used on an HPC system. It uses a distributed MACE calculator. For non-distributed calculations, users may replace the distributed calculator section with a standard ASE-compatible MACE calculator or another ASE-compatible MLIP calculator.

## How to run the MC–MD workflow
````markdown
### Step 1: Prepare the GB template

Place the provided Σ5(210) GB template in the repository root and name it:

```text
initial.xyz
Example for (Hf,Mo,V,W,Zr)C:
python scripts/make_sigma5_210_gb_from_initial.py \
  --template initial.xyz \
  --composition "Hf Mo V W Zr" \
  --seed 12345 \
  --orthogonalize-cell \
  --output HfMoVWZrC_sigma5_210_GB.xyz \
  --write-lammps-data HfMoVWZrC_sigma5_210_GB.data
```

This produces:

```text
HfMoVWZrC_sigma5_210_GB.xyz
HfMoVWZrC_sigma5_210_GB.data
```
The generated `.xyz` file should be copied into the temperature run folder and renamed as `initial.xyz`.

### Step 2: Prepare the model

Place the public pretrained MACE-OMAT-0 small model in the same folder and name it:

small.model

Alternatively, edit the model_paths variable in the MC–MD script.

### Step 3: Use a temperature folder name

The script reads the target temperature automatically from the current folder name.

Accepted folder-name examples include:

300
300K
T300
temp300
2000
2000K
T2000
temp2000

For example, to run a 300 K simulation, run the script inside a folder named:

300K
### Step 4: Run the script

Example command:

python scripts/mc_md_workflow.py

or, if running from inside a temperature folder:

python ../scripts/mc_md_workflow.py
Main MC–MD settings

Important MC–MD settings are defined near the top of the script:

swap_steps
initial_relax_steps
short_md_steps
timestep_fs
snapshot_interval
snapshot_window
non_swappable_species

In the manuscript workflow, carbon was treated as non-swappable and metal atoms were allowed to swap.

MC–MD output files

The MC–MD script writes:

all_swaps.csv
accepted_swaps.csv
md_summary.csv
restart_state.json
best_structure.xyz
run_info.txt
accepted_snapshots/*.xyz

If both restart_state.json and best_structure.xyz are present, the script resumes from the previous MC step.

## How to run the GB segregation post-processing
### Step 1: Prepare .xyz files

Place the .xyz files to be analyzed in the folder where the post-processing script will be run.

The script automatically processes all files ending with:

.xyz

in the current directory.

### Step 2: Run the script

From the folder containing the .xyz files, run:

python path/to/scripts/gb_segregation_postprocessing.py

Example:

python ../scripts/gb_segregation_postprocessing.py

Important note: the script processes all .xyz files in the current directory. Therefore, run it in a folder containing only the .xyz files intended for analysis.

## GB segregation post-processing outputs

For each input .xyz file, the post-processing script writes:

*_gb_metals_normalized_equalcount.csv
*_gb_metals_normalized_equalcount.png
*_gb_metals_normalized_equalcount.pdf

The script also writes two summary files:

gb_top_segregant_selectivity_summary.csv
gb_all_species_ranked_enrichment.csv

The summary files report the dominant GB species, second-highest GB species, GB composition, bulk composition, enrichment relative to the bulk region, and segregation selectivity.

## Binning and GB-core definition

By default, the post-processing script uses equal-count distance bins:

EQUAL_COUNT = True

The GB region used for top-segregant analysis is controlled by:

GB_MODE = "first_n_bins"
FIRST_N_BINS = 2

Therefore, by default, the first two bins closest to the GB are used as the GB-core region.

The bulk region is selected from the farthest part of the distance profile using:

BULK_FRACTION_START = 0.75

These settings can be modified directly in the script.



## Reproducibility scope

This repository provides the scripts and workflow documentation used for MC–MD sampling and GB segregation post-processing.

Users should provide their own:

input grain-boundary structures;
pretrained model file or ASE-compatible calculator;
run folders;
.xyz files for post-processing;
output directories.

The scripts may need minor path, calculator, or geometry modifications depending on the user’s computing environment, input structure format, and GB setup.

## Citation

If this repository is used, please cite the associated manuscript and the archived repository DOI when available.


