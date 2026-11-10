#!/usr/bin/env python3
"""Build WT/mutant T4 lysozyme systems and run matched OpenMM MD.

The fast pilot uses GBn2 implicit solvent.  The same script also supports
explicit TIP3P-FB solvent for longer production replicas.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, unit
from openmm.app import (
    ForceField,
    HBonds,
    Modeller,
    NoCutoff,
    PME,
    PDBFile,
    Simulation,
    StateDataReporter,
    XTCReporter,
)
from pdbfixer import PDBFixer


VARIANTS = {
    "wild_type": None,
    "stabilizing_S117V": "SER-117-VAL",
    "destabilizing_G30F": "GLY-30-PHE",
}

if os.environ.get("OPENMM_PLUGIN_DIR"):
    Platform.loadPluginsFromDirectory(os.environ["OPENMM_PLUGIN_DIR"])


def platform_candidates(name: str):
    available = {
        Platform.getPlatform(index).getName()
        for index in range(Platform.getNumPlatforms())
    }
    if name != "auto":
        if name not in available:
            raise ValueError(
                f"Platform {name!r} is unavailable; found {sorted(available)}"
            )
        names = [name]
    else:
        names = [candidate for candidate in ("CUDA", "OpenCL", "CPU", "Reference")
                 if candidate in available]
    for candidate in names:
        platform = Platform.getPlatformByName(candidate)
        properties = (
            {"Precision": "mixed"} if candidate in {"CUDA", "OpenCL"} else {}
        )
        yield platform, properties


def build_model(source_pdb: Path, mutation: str | None, ph: float):
    fixer = PDBFixer(filename=str(source_pdb))
    if mutation is not None:
        fixer.applyMutations([mutation], "A")
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    return Modeller(fixer.topology, fixer.positions)


def write_pdb(path: Path, topology, positions) -> None:
    with path.open("w") as handle:
        PDBFile.writeFile(topology, positions, handle, keepIds=True)


def run_variant(args, name: str, mutation: str | None) -> dict:
    output_dir = args.output_dir / name / f"replica_{args.replica}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.xtc"
    if trajectory_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{trajectory_path} already exists. Use --overwrite only if replacement is intended."
        )

    started = time.time()
    modeller = build_model(args.pdb, mutation, args.ph)

    if args.solvent == "implicit":
        forcefield_files = ["amber14-all.xml", "implicit/gbn2.xml"]
        forcefield = ForceField(*forcefield_files)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
        )
        ensemble = "NVT"
    else:
        forcefield_files = ["amber14-all.xml", "amber14/tip3pfb.xml"]
        forcefield = ForceField(*forcefield_files)
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=args.padding_nm * unit.nanometer,
            ionicStrength=args.ionic_strength_molar * unit.molar,
            positiveIon="Na+",
            negativeIon="Cl-",
        )
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=args.cutoff_nm * unit.nanometer,
            constraints=HBonds,
        )
        system.addForce(
            MonteCarloBarostat(
                args.pressure_bar * unit.bar,
                args.temperature_k * unit.kelvin,
                25,
            )
        )
        ensemble = "NPT"

    write_pdb(output_dir / "prepared.pdb", modeller.topology, modeller.positions)

    integrator = LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        args.friction_per_ps / unit.picosecond,
        args.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed + args.replica)
    failures = []
    simulation = None
    platform = None
    for candidate, properties in platform_candidates(args.platform):
        try:
            simulation = Simulation(
                modeller.topology,
                system,
                integrator,
                candidate,
                properties,
            )
            platform = candidate
            break
        except Exception as exc:
            failures.append(f"{candidate.getName()}: {exc}")
            if args.platform != "auto":
                raise
    if simulation is None or platform is None:
        raise RuntimeError("No OpenMM platform could create a context: " + "; ".join(failures))
    if failures:
        print(
            "Platform fallback: " + "; ".join(failures)
            + f"; using {platform.getName()}",
            flush=True,
        )
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
    simulation.context.setVelocitiesToTemperature(
        args.temperature_k * unit.kelvin,
        args.seed + args.replica,
    )

    equilibration_steps = round(
        args.equilibration_ps * 1000.0 / args.timestep_fs
    )
    production_steps = round(args.production_ps * 1000.0 / args.timestep_fs)
    report_steps = max(1, round(args.report_interval_ps * 1000.0 / args.timestep_fs))
    simulation.step(equilibration_steps)

    equilibrated = simulation.context.getState(getPositions=True)
    write_pdb(
        output_dir / "equilibrated.pdb",
        modeller.topology,
        equilibrated.getPositions(),
    )

    simulation.reporters.append(XTCReporter(str(trajectory_path), report_steps))
    state_kwargs = dict(
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
        separator=",",
    )
    if args.solvent == "explicit":
        state_kwargs.update(volume=True, density=True)
    simulation.reporters.append(
        StateDataReporter(str(output_dir / "state.csv"), report_steps, **state_kwargs)
    )
    simulation.reporters.append(
        StateDataReporter(
            sys.stdout,
            max(report_steps, production_steps // 10),
            step=True,
            time=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            totalSteps=production_steps,
            separator="\t",
        )
    )
    simulation.step(production_steps)
    simulation.saveCheckpoint(str(output_dir / "final.chk"))
    final_state = simulation.context.getState(
        getPositions=True,
        getEnergy=True,
    )
    write_pdb(output_dir / "final.pdb", modeller.topology, final_state.getPositions())

    atom_count = modeller.topology.getNumAtoms()
    residue_count = modeller.topology.getNumResidues()
    metadata = {
        "variant": name,
        "mutation": mutation or "WT",
        "replica": args.replica,
        "source_pdb": str(args.pdb.resolve()),
        "pH_used_for_hydrogen_placement": args.ph,
        "force_field_files": forcefield_files,
        "solvent": args.solvent,
        "ensemble": ensemble,
        "temperature_K": args.temperature_k,
        "pressure_bar": args.pressure_bar if args.solvent == "explicit" else None,
        "ionic_strength_M": (
            args.ionic_strength_molar if args.solvent == "explicit" else None
        ),
        "padding_nm": args.padding_nm if args.solvent == "explicit" else None,
        "timestep_fs": args.timestep_fs,
        "friction_per_ps": args.friction_per_ps,
        "minimize_iterations": args.minimize_iterations,
        "equilibration_ps": args.equilibration_ps,
        "production_ps": args.production_ps,
        "report_interval_ps": args.report_interval_ps,
        "seed": args.seed + args.replica,
        "platform": platform.getName(),
        "atoms": atom_count,
        "residues": residue_count,
        "production_steps": production_steps,
        "trajectory_frames_expected": production_steps // report_steps,
        "wall_seconds": time.time() - started,
        "final_potential_energy_kj_mol": (
            final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ),
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(
        f"Completed {name} replica {args.replica}: "
        f"{metadata['trajectory_frames_expected']} frames, "
        f"{metadata['wall_seconds']:.1f} s",
        flush=True,
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=Path, default=Path(__file__).with_name("2LZM.pdb"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("runs"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--solvent", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--ph", type=float, default=5.4)
    parser.add_argument("--ionic-strength-molar", type=float, default=0.15)
    parser.add_argument("--padding-nm", type=float, default=1.0)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--minimize-iterations", type=int, default=1000)
    parser.add_argument("--equilibration-ps", type=float, default=20.0)
    parser.add_argument("--production-ps", type=float, default=100.0)
    parser.add_argument("--report-interval-ps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--platform",
        choices=["auto", "CUDA", "OpenCL", "CPU", "Reference"],
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replica < 0:
        raise ValueError("--replica must be non-negative")
    summaries = [
        run_variant(args, name, VARIANTS[name])
        for name in args.variants
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
