#!/usr/bin/env python3
"""
Monte Carlo Atom Swap with Distributed MACE + reduced file output

What this version does
----------------------
1. Reads species automatically from initial.xyz
2. Reads target temperature from folder name:
      300     or   300K   or   T300   or   temp300
3. Saves sparse accepted snapshots only:
      - step_0
      - then one accepted structure around every 10 MC steps
      - if no accepted structure exists near that checkpoint,
        it saves the first accepted structure after that checkpoint
4. Keeps file count low:
      - all_swaps.csv
      - accepted_swaps.csv
      - md_summary.csv
      - restart_state.json
      - best_structure.xyz
      - accepted_snapshots/*.xyz
"""

import os
import re
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from ase import units
from ase.data import atomic_numbers
from ase.io import read, write as ase_write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# ------------------------------------------------------------
# NPT IMPORTS (FULL MTK ONLY, NO FALLBACK)
# ------------------------------------------------------------
try:
    from ase.md.nose_hoover_chain import MTKNPT as MTK_CLASS
    print("✅ Using FULL MTK: ase.md.nose_hoover_chain.MTKNPT")
except Exception as e:
    raise ImportError(
        "MTKNPT import failed. You requested MTK-only (no fallback).\n"
        f"Error: {e}"
    )

from DistMLIP.implementations.mace.mace import MACECalculator_Dist
from mace.calculators import MACECalculator

# ============================================================
# CONFIG
# ============================================================
initial_file = "initial.xyz"

# MC / MD settings
swap_steps = 100000
initial_relax_steps = 1500
pe_avg_window_init = 150
short_md_steps = 10
timestep_fs = 1.5
k_B = 8.617333262145e-5  # eV/K

# sparse snapshot settings
snapshot_interval = 10      # target every 10 MC steps
snapshot_window = 5         # "around that" window = +/- 5 steps

# species behavior
# If you want all species to be swappable, set this to an empty set().
non_swappable_species = {"C"}

# files / folders
snapshot_dir = "accepted_snapshots"
state_file = "restart_state.json"
candidate_file = "snapshot_candidate.xyz"

os.makedirs(snapshot_dir, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def get_temperature_from_folder():
    """Read temperature from current folder name."""
    folder = os.path.basename(os.getcwd()).strip()

    patterns = [
        r'^(\d+(?:\.\d+)?)$',                 # 300
        r'^(\d+(?:\.\d+)?)\s*[Kk]$',          # 300K
        r'^[Tt]_?(\d+(?:\.\d+)?)$',           # T300 or T_300
        r'^[Tt]emp[_\-]?(\d+(?:\.\d+)?)$',    # temp300 or temp_300
    ]

    for pat in patterns:
        m = re.match(pat, folder)
        if m:
            return float(m.group(1))

    raise ValueError(
        f"Could not parse temperature from folder name '{folder}'. "
        f"Use names like 300, 300K, T300, or temp300."
    )


def append_row(filename, row_dict):
    """Append one row to CSV, auto-create header if file does not exist."""
    header = not os.path.exists(filename)
    with open(filename, "a") as f:
        pd.DataFrame([row_dict]).to_csv(f, index=False, header=header)


def set_atomic_numbers(atoms):
    atoms.set_atomic_numbers([atomic_numbers[s] for s in atoms.get_chemical_symbols()])


def save_json_state(state):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def load_json_state():
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return None


def save_snapshot(target_step, source_step, atoms):
    """Save sparse accepted structure."""
    fname = os.path.join(
        snapshot_dir,
        f"snapshot_target_{target_step:06d}_from_step_{source_step:06d}.xyz"
    )
    ase_write(fname, atoms)
    print(f"[SNAPSHOT] saved: {fname}")


def write_run_info(all_species, swappable_species, temperature):
    with open("run_info.txt", "w") as f:
        f.write(f"initial_file = {initial_file}\n")
        f.write(f"all_species_in_xyz = {all_species}\n")
        f.write(f"non_swappable_species = {sorted(non_swappable_species)}\n")
        f.write(f"swappable_species = {swappable_species}\n")
        f.write(f"T_target_K = {temperature}\n")
        f.write(f"swap_steps = {swap_steps}\n")
        f.write(f"initial_relax_steps = {initial_relax_steps}\n")
        f.write(f"short_md_steps = {short_md_steps}\n")
        f.write(f"snapshot_interval = {snapshot_interval}\n")
        f.write(f"snapshot_window = {snapshot_window}\n")


# ============================================================
# READ SPECIES FROM XYZ
# ============================================================
initial_atoms = read(initial_file)
all_species = sorted(set(initial_atoms.get_chemical_symbols()))
elements_to_swap = [s for s in all_species if s not in non_swappable_species]

if len(elements_to_swap) < 2:
    raise ValueError(
        f"Need at least 2 swappable species. Found swappable species: {elements_to_swap}"
    )

print("Species found in XYZ:", all_species)
print("Swappable species:", elements_to_swap)

# ============================================================
# READ TEMPERATURE FROM FOLDER NAME
# ============================================================
T_target = get_temperature_from_folder()
print(f"Target temperature from folder name: {T_target} K")

write_run_info(all_species, elements_to_swap, T_target)

# ============================================================
# GPU / CALCULATOR SETUP
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gpu_ids = [0, 1]

print("Using GPUs:", gpu_ids)
for i in gpu_ids:
    print(f"  - {torch.cuda.get_device_name(i)}")

base_calc = MACECalculator(
    model_paths="small.model",
    device="cuda:0",
    head="omat_pbe"
)

dist_calc = MACECalculator_Dist.from_existing(base_calc)
dist_calc.enable_distributed_mode(gpu_ids)

# ============================================================
# MD DYNAMICS
# ============================================================
def make_dynamics(atoms):
    """Return MTK NPT only. No fallback thermostat/barostat."""
    return MTK_CLASS(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=T_target,
        pressure_au=0.0,
        tdamp=100 * units.fs,
        pdamp=1000 * units.fs,
    )


def run_md_relax(atoms, nsteps, label, mc_step, avg_window=None):
    """
    Run MD relaxation with reduced logging:
    only one line in md_summary.csv per MD run.
    """
    atoms.calc = dist_calc

    # Initialize velocities only if missing (important for restart safety)
    if not atoms.has("velocities"):
        MaxwellBoltzmannDistribution(atoms, temperature_K=T_target)

    dyn = make_dynamics(atoms)

    pe_history = []
    T_history = []

    t0 = time.time()
    for _ in range(nsteps):
        dyn.run(1)
        pe_history.append(atoms.get_potential_energy())
        T_history.append(atoms.get_temperature())
    elapsed = time.time() - t0

    if avg_window is not None:
        if len(pe_history) >= avg_window:
            avg_pe = float(np.mean(pe_history[-avg_window:]))
            avg_T = float(np.mean(T_history[-avg_window:]))
        else:
            avg_pe = float(np.mean(pe_history))
            avg_T = float(np.mean(T_history))
    else:
        avg_pe = float(pe_history[-1])
        avg_T = float(T_history[-1])

    final_pe = float(pe_history[-1])
    final_T = float(T_history[-1])

    append_row("md_summary.csv", {
        "MC_Step": mc_step,
        "Label": label,
        "MD_Steps": nsteps,
        "Avg_PE(eV)": avg_pe,
        "Final_PE(eV)": final_pe,
        "Avg_Temp(K)": avg_T,
        "Final_Temp(K)": final_T,
        "Elapsed_Time(s)": round(elapsed, 4)
    })

    return avg_pe, atoms.copy(), elapsed


# ============================================================
# SWAP
# ============================================================
def perform_swap(atoms, seen_pairs):
    indices = [i for i, a in enumerate(atoms) if a.symbol in elements_to_swap]

    while True:
        i1, i2 = random.sample(indices, 2)
        if atoms[i1].symbol == atoms[i2].symbol:
            continue
        key = tuple(sorted((i1, i2)))
        if key not in seen_pairs:
            seen_pairs.add(key)
            break

    sym1_before = atoms[i1].symbol
    sym2_before = atoms[i2].symbol

    atoms[i1].symbol, atoms[i2].symbol = atoms[i2].symbol, atoms[i1].symbol
    return i1, i2, sym1_before, sym2_before, atoms.copy()


# ============================================================
# SPARSE SNAPSHOT LOGIC
# ============================================================
def maybe_update_candidate(candidate, target_step, accepted_step, atoms):
    """
    Keep the accepted structure closest to the current target step
    within [target - window, target + window].
    """
    lo = target_step - snapshot_window
    hi = target_step + snapshot_window

    if lo <= accepted_step <= hi:
        dist = abs(accepted_step - target_step)
        if (
            candidate is None
            or dist < candidate["dist"]
            or (dist == candidate["dist"] and accepted_step < candidate["step"])
        ):
            ase_write(candidate_file, atoms)
            candidate = {"step": accepted_step, "dist": dist}
            print(f"[SNAPSHOT] candidate updated for target {target_step}: step {accepted_step}")
    return candidate


def maybe_finalize_due_target(current_mc_step, next_target, candidate, waiting_for_future_accept):
    """
    Once we are past target + window:
      - if candidate exists, save it
      - if no candidate exists, wait for next accepted structure after target
    """
    while current_mc_step > next_target + snapshot_window:
        if candidate is not None:
            cand_atoms = read(candidate_file)
            save_snapshot(next_target, candidate["step"], cand_atoms)

            candidate = None
            if os.path.exists(candidate_file):
                os.remove(candidate_file)

            next_target += snapshot_interval
            waiting_for_future_accept = False
            continue

        if waiting_for_future_accept:
            break

        waiting_for_future_accept = True
        print(f"[SNAPSHOT] no nearby accepted structure for target {next_target}; waiting for next accepted step")
        break

    return next_target, candidate, waiting_for_future_accept


def satisfy_waiting_target_if_needed(step, next_target, waiting_for_future_accept, atoms):
    """
    If no accepted structure existed near the target, save the first
    accepted structure after that target as the representative one.
    """
    if waiting_for_future_accept and step > next_target + snapshot_window:
        save_snapshot(next_target, step, atoms)
        next_target += snapshot_interval
        waiting_for_future_accept = False
        print(f"[SNAPSHOT] used future accepted step {step} for missed target")

    return next_target, waiting_for_future_accept


# ============================================================
# RESTART / INITIALIZATION
# ============================================================
state = load_json_state()

candidate = None
next_snapshot_target = snapshot_interval
waiting_for_future_accept = False

if state is not None and os.path.exists("best_structure.xyz"):
    current_step = int(state["current_step"])
    best_energy = float(state["best_energy"])
    next_snapshot_target = int(state.get("next_snapshot_target", snapshot_interval))
    waiting_for_future_accept = bool(state.get("waiting_for_future_accept", False))

    best_atoms = read("best_structure.xyz")
    set_atomic_numbers(best_atoms)

    cand_step = state.get("candidate_step", None)
    cand_dist = state.get("candidate_dist", None)
    if cand_step is not None and os.path.exists(candidate_file):
        candidate = {"step": int(cand_step), "dist": float(cand_dist)}

    print(f"[RESTART] continuing from step {current_step}")
    print(f"[RESTART] best_energy = {best_energy:.6f} eV")
    print(f"[RESTART] next_snapshot_target = {next_snapshot_target}")
else:
    current_step = 0
    best_atoms = initial_atoms.copy()
    set_atomic_numbers(best_atoms)

    print("[INIT] initial NPT relaxation ...")
    best_energy, best_atoms, init_time = run_md_relax(
        best_atoms,
        initial_relax_steps,
        label="init",
        mc_step=0,
        avg_window=pe_avg_window_init
    )

    ase_write("best_structure.xyz", best_atoms)

    # save initial snapshot
    save_snapshot(0, 0, best_atoms)

    append_row("all_swaps.csv", {
        "Step": 0,
        "Atom1_ID": -1,
        "Atom2_ID": -1,
        "Element_1": "None",
        "Element_2": "None",
        "Trial_PE_BeforeMD(eV)": best_energy,
        "Reference_Best_PE_BeforeDecision(eV)": best_energy,
        "DeltaE_TrialMinusBest(eV)": 0.0,
        "Decision": "Initial",
        "Accepted": True,
        "MD_Time(s)": round(init_time, 4),
        "Best_PE_AfterMD(eV)": best_energy
    })

    append_row("accepted_swaps.csv", {
        "Step": 0,
        "Atom1_ID": -1,
        "Atom2_ID": -1,
        "Element_1": "None",
        "Element_2": "None",
        "Trial_PE_BeforeMD(eV)": best_energy,
        "Reference_Best_PE_BeforeDecision(eV)": best_energy,
        "DeltaE_TrialMinusBest(eV)": 0.0,
        "Decision": "Initial",
        "Accepted": True,
        "MD_Time(s)": round(init_time, 4),
        "Best_PE_AfterMD(eV)": best_energy
    })

    save_json_state({
        "current_step": current_step,
        "best_energy": best_energy,
        "next_snapshot_target": next_snapshot_target,
        "waiting_for_future_accept": waiting_for_future_accept,
        "candidate_step": None,
        "candidate_dist": None
    })

# ============================================================
# MONTE CARLO LOOP
# ============================================================
seen_pairs = set()

for step in range(current_step + 1, swap_steps + 1):
    print(f"\n[MC] Step {step}")

    i1, i2, sym1, sym2, trial_atoms = perform_swap(best_atoms.copy(), seen_pairs)
    set_atomic_numbers(trial_atoms)

    # carry velocities from best structure if available
    if best_atoms.has("velocities"):
        trial_atoms.set_velocities(best_atoms.get_velocities().copy())

    # single-point energy before decision
    try:
        trial_atoms.calc = dist_calc
        t0 = time.time()
        trial_pe = float(trial_atoms.get_potential_energy())
        sp_time = time.time() - t0
    except Exception as e:
        print(f"[ERROR] step {step}: single-point energy failed: {e}")
        continue

    dE = trial_pe - best_energy

    if trial_pe < best_energy:
        accept = True
        decision = "Accepted (lower)"
    else:
        p = np.exp(-dE / (k_B * T_target))
        accept = random.random() < p
        decision = f"Accepted (MC p={p:.6f})" if accept else f"Rejected (MC p={p:.6f})"

    row = {
        "Step": step,
        "Atom1_ID": i1,
        "Atom2_ID": i2,
        "Element_1": sym1,
        "Element_2": sym2,
        "Trial_PE_BeforeMD(eV)": trial_pe,
        "Reference_Best_PE_BeforeDecision(eV)": best_energy,
        "DeltaE_TrialMinusBest(eV)": dE,
        "Decision": decision,
        "Accepted": accept,
        "MD_Time(s)": None,
        "Best_PE_AfterMD(eV)": None
    }

    if accept:
        best_energy, best_atoms, md_time = run_md_relax(
            trial_atoms.copy(),
            short_md_steps,
            label=f"accept_{step}",
            mc_step=step,
            avg_window=None
        )

        row["MD_Time(s)"] = round(md_time, 4)
        row["Best_PE_AfterMD(eV)"] = best_energy

        ase_write("best_structure.xyz", best_atoms)
        append_row("accepted_swaps.csv", row)

        # if a target was missed, use this first future accepted structure
        next_snapshot_target, waiting_for_future_accept = satisfy_waiting_target_if_needed(
            step, next_snapshot_target, waiting_for_future_accept, best_atoms
        )

        # then see whether this same accepted step is candidate for the current target
        candidate = maybe_update_candidate(candidate, next_snapshot_target, step, best_atoms)

    else:
        # keep current best structure and just do short MD on it
        best_energy, best_atoms, md_time = run_md_relax(
            best_atoms.copy(),
            short_md_steps,
            label=f"reject_{step}",
            mc_step=step,
            avg_window=None
        )

        row["MD_Time(s)"] = round(md_time, 4)
        row["Best_PE_AfterMD(eV)"] = best_energy

        ase_write("best_structure.xyz", best_atoms)

    append_row("all_swaps.csv", row)

    # if we are sufficiently past a target, finalize it
    next_snapshot_target, candidate, waiting_for_future_accept = maybe_finalize_due_target(
        step, next_snapshot_target, candidate, waiting_for_future_accept
    )

    # save restart state every step
    save_json_state({
        "current_step": step,
        "best_energy": best_energy,
        "next_snapshot_target": next_snapshot_target,
        "waiting_for_future_accept": waiting_for_future_accept,
        "candidate_step": None if candidate is None else candidate["step"],
        "candidate_dist": None if candidate is None else candidate["dist"]
    })

print(f"\nFinished MC loop. Final best PE = {best_energy:.6f} eV")

