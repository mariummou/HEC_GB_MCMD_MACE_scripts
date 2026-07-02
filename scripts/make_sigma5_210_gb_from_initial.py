#!/usr/bin/env python3
"""
Build HEC rocksalt Sigma5(210)<001> GB supercells by using an existing GB XYZ file
as the exact geometry/template and only replacing the metal-sublattice species.

This is the safest way to reproduce the same GB structural motif as the reference
file, including the microscopic GB translation/termination.  The coordinates and
cell are kept unchanged; only non-C atoms are randomized among the requested
metal species.

Example:
  python gb_from_reference_template.py \
    --template 300_low.xyz \
    --composition "Hf Mo V W Zr" \
    --seed 12345 \
    --output HfMoVWZrC_sigma5_template.xyz \
    --write-lammps-data HfMoVWZrC_sigma5_template.data

Make the six compositions:
  python gb_from_reference_template.py --template 300_low.xyz --make-six --seed 12345
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write


SIX_HEC_COMPOSITIONS = {
    "HfNbTaTiZrC": ["Hf", "Nb", "Ta", "Ti", "Zr"],
    "HfMoNbTaZrC": ["Hf", "Mo", "Nb", "Ta", "Zr"],
    "NbTaTiWZrC": ["Nb", "Ta", "Ti", "W", "Zr"],
    "HfTaTiWZrC": ["Hf", "Ta", "Ti", "W", "Zr"],
    "CrHfTaTiZrC": ["Cr", "Hf", "Ta", "Ti", "Zr"],
    "HfMoVWZrC": ["Hf", "Mo", "V", "W", "Zr"],
}

ALIASES = {
    "low": ["Hf", "Mo", "V", "W", "Zr"],
    "HfMoVWZr": ["Hf", "Mo", "V", "W", "Zr"],
    "HfMoVWZrC": ["Hf", "Mo", "V", "W", "Zr"],
    "NbTaTiWZr": ["Nb", "Ta", "Ti", "W", "Zr"],
    "NbTaTiWZrC": ["Nb", "Ta", "Ti", "W", "Zr"],
    "HfTaTiWZr": ["Hf", "Ta", "Ti", "W", "Zr"],
    "HfTaTiWZrC": ["Hf", "Ta", "Ti", "W", "Zr"],
    "HfNbTaTiZr": ["Hf", "Nb", "Ta", "Ti", "Zr"],
    "HfNbTaTiZrC": ["Hf", "Nb", "Ta", "Ti", "Zr"],
    "HfMoNbTaZr": ["Hf", "Mo", "Nb", "Ta", "Zr"],
    "HfMoNbTaZrC": ["Hf", "Mo", "Nb", "Ta", "Zr"],
    "CrHfTaTiZr": ["Cr", "Hf", "Ta", "Ti", "Zr"],
    "CrHfTaTiZrC": ["Cr", "Hf", "Ta", "Ti", "Zr"],
}


def parse_metals(text: str) -> list[str]:
    """Accept 'Hf Mo V W Zr', 'Hf,Mo,V,W,Zr', or '(Hf, Mo, V, W, Zr)C'."""
    key = text.strip()
    if key in ALIASES:
        return ALIASES[key]

    clean = key.replace("(", "").replace(")", "")
    clean = re.sub(r"C\s*$", "", clean.strip())
    parts = [p for p in re.split(r"[,;+\s]+", clean) if p]
    if len(parts) < 2:
        raise ValueError("Could not parse composition. Example: --composition 'Hf Mo V W Zr'")
    for p in parts:
        if p not in atomic_numbers:
            raise ValueError(f"Unknown element symbol: {p}")
    return parts


def composition_label(metals: list[str], pretty: bool = False) -> str:
    if pretty:
        return "(" + ",".join(metals) + ")C"
    return "".join(metals) + "C"


def assign_equimolar_metals(template_atoms, metals: list[str], seed: int):
    atoms = template_atoms.copy()
    atoms.pbc = (True, True, True)
    atoms.wrap(eps=1.0e-12)

    symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
    c_mask = symbols == "C"
    metal_idx = np.where(~c_mask)[0]
    n_metal = len(metal_idx)

    if n_metal == 0:
        raise ValueError("No metal sites found. The template must contain C and metal atoms.")

    base = n_metal // len(metals)
    rem = n_metal % len(metals)
    target_counts = {m: base + (i < rem) for i, m in enumerate(metals)}

    labels = []
    for m in metals:
        labels.extend([m] * target_counts[m])
    labels = np.array(labels, dtype=object)

    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    symbols[metal_idx] = labels

    atoms.set_chemical_symbols(symbols.tolist())
    atoms.set_masses([atomic_masses[atomic_numbers[s]] for s in symbols])

    # Useful for LAMMPS/OVITO; ASE extxyz preserves this as an auxiliary property.
    if "id" in atoms.arrays:
        del atoms.arrays["id"]
    atoms.new_array("id", np.arange(1, len(atoms) + 1, dtype=np.int32))
    return atoms, target_counts


def summarize_template(atoms):
    symbols = atoms.get_chemical_symbols()
    lengths = atoms.cell.lengths()
    n_c = sum(s == "C" for s in symbols)
    n_metal = len(atoms) - n_c
    # For your Sigma5 geometry, using nx=8, ny=4, nz=12:
    a_from_lx = lengths[0] / (8.0 * math.sqrt(5.0))
    a_from_ly = lengths[1] / (4.0 * math.sqrt(5.0))
    a_from_lz = lengths[2] / 12.0
    return {
        "n_total": len(atoms),
        "n_c": n_c,
        "n_metal": n_metal,
        "lengths": lengths,
        "a_estimates": (a_from_lx, a_from_ly, a_from_lz),
        "input_counts": Counter(symbols),
    }



def clean_box_for_ovito(atoms, orthogonalize: bool = False):
    """Make the extxyz cell display cleanly in OVITO/LAMMPS.

    The reference XYZ may contain an Origin such as (0,0,-18).  ASE's wrap()
    wraps coordinates around a zero-origin cell, but extxyz would otherwise keep
    the old Origin metadata.  OVITO then draws the simulation box at the old
    origin while the atoms are wrapped around zero, making the box look shifted
    or weird.  Removing Origin fixes that display mismatch.

    If orthogonalize=True, also replace a tiny triclinic/sheared cell by a
    rectangular box with the same cell-vector lengths.  This is only for clean
    visualization / orthogonal LAMMPS data; coordinates are not scaled.
    """
    # Remove old extxyz origin metadata so OVITO draws the box around wrapped atoms.
    for key in ("Origin", "origin"):
        if key in atoms.info:
            del atoms.info[key]

    if orthogonalize:
        lengths = atoms.cell.lengths()
        pos = atoms.positions.copy()
        atoms.set_cell(np.diag(lengths), scale_atoms=False)
        atoms.positions[:] = pos

    atoms.pbc = (True, True, True)
    atoms.wrap(eps=1.0e-12)
    return atoms


def write_one(template, metals: list[str], seed: int, output: str | None, data_file: str | None, orthogonalize_cell: bool = False):
    atoms, metal_counts = assign_equimolar_metals(template, metals, seed)
    clean_box_for_ovito(atoms, orthogonalize=orthogonalize_cell)
    comp = composition_label(metals, pretty=False)
    if output is None:
        output = f"{comp}_sigma5_210_templateGB.xyz"
    write(output, atoms, format="extxyz")

    if data_file:
        specorder = ["C"] + metals
        write(data_file, atoms, format="lammps-data", atom_style="atomic", masses=True, specorder=specorder)

    lengths = atoms.cell.lengths()
    print("Built:", composition_label(metals, pretty=True))
    print(f"  output: {output}")
    if data_file:
        print(f"  data:   {data_file}")
    print(f"  atoms: total={len(atoms)}, C={sum(s == 'C' for s in atoms.get_chemical_symbols())}, metals={len(atoms) - sum(s == 'C' for s in atoms.get_chemical_symbols())}")
    print(f"  metal counts: {metal_counts}")
    print(f"  cell: Lx={lengths[0]:.6f} A, Ly={lengths[1]:.6f} A, Lz={lengths[2]:.6f} A")
    print(f"  GB planes under PBC: x=0/Lx and x=Lx/2={lengths[0] / 2.0:.6f} A")
    return output


def main():
    ap = argparse.ArgumentParser(description="Use an existing Sigma5(210)<001> GB XYZ as geometry template and assign HEC metal species.")
    ap.add_argument("--template", "--reference-xyz", dest="template", required=True,
                    help="Existing GB xyz/extxyz file whose coordinates/cell are kept exactly, e.g. 300_low.xyz")
    ap.add_argument("--composition", default="Hf Mo V W Zr",
                    help="Metals to place on the metal sublattice, e.g. 'Hf Mo V W Zr' or '(Nb,Ta,Ti,W,Zr)C'. Ignored with --make-six.")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for metal assignment.")
    ap.add_argument("--output", default=None, help="Output xyz/extxyz file. Not used with --make-six.")
    ap.add_argument("--write-lammps-data", default=None, help="Optional LAMMPS data file. Not used with --make-six unless --data-dir is set.")
    ap.add_argument("--make-six", action="store_true", help="Generate the six HEC compositions used in the paper.")
    ap.add_argument("--orthogonalize-cell", action="store_true",
                    help="Replace the tiny sheared/triclinic template cell by a rectangular box with same lengths. Also clears Origin metadata. Good for clean OVITO display and orthogonal LAMMPS data.")
    ap.add_argument("--outdir", default=".", help="Output directory for --make-six or default output.")
    ap.add_argument("--data-dir", default=None, help="With --make-six, also write LAMMPS data files into this directory.")
    args = ap.parse_args()

    template = read(args.template)
    template.pbc = (True, True, True)
    template.wrap(eps=1.0e-12)
    clean_box_for_ovito(template, orthogonalize=args.orthogonalize_cell)

    info = summarize_template(template)
    print("Template:", args.template)
    print(f"  atoms: total={info['n_total']}, C={info['n_c']}, metals={info['n_metal']}")
    print(f"  input counts: {dict(info['input_counts'])}")
    print(f"  cell: Lx={info['lengths'][0]:.6f} A, Ly={info['lengths'][1]:.6f} A, Lz={info['lengths'][2]:.6f} A")
    print(f"  a estimates from Sigma5 repeats: {info['a_estimates'][0]:.6f}, {info['a_estimates'][1]:.6f}, {info['a_estimates'][2]:.6f} A")
    print("  geometry mode: exact template coordinates are preserved; only metal symbols are changed")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if args.make_six:
        if args.data_dir:
            Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        for i, (name, metals) in enumerate(SIX_HEC_COMPOSITIONS.items()):
            out = str(Path(args.outdir) / f"{name}_sigma5_210_templateGB.xyz")
            data = str(Path(args.data_dir) / f"{name}_sigma5_210_templateGB.data") if args.data_dir else None
            # Offset the seed so each composition has a different random ordering.
            write_one(template, metals, args.seed + i, out, data, orthogonalize_cell=args.orthogonalize_cell)
        return

    metals = parse_metals(args.composition)
    out = args.output
    if out is None:
        out = str(Path(args.outdir) / f"{composition_label(metals)}_sigma5_210_templateGB.xyz")
    data = args.write_lammps_data
    if data is not None:
        data = str(Path(data))
    write_one(template, metals, args.seed, out, data, orthogonalize_cell=args.orthogonalize_cell)


if __name__ == "__main__":
    main()
