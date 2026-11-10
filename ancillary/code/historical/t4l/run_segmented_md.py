#!/usr/bin/env python3
"""Run one interruption-resilient segment of a long OpenMM trajectory.

Each replica is prepared and equilibrated once.  Subsequent invocations restore
the same serialized system and latest binary checkpoint, append trajectory
frames, and advance to the next fixed segment boundary.  A frequently replaced
checkpoint limits lost work if a node or wall-time interruption occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    XmlSerializer,
    unit,
)
from openmm.app import (
    CheckpointReporter,
    ForceField,
    HBonds,
    Modeller,
    PME,
    PDBFile,
    Simulation,
    StateDataReporter,
    XTCReporter,
)

from run_openmm_md import VARIANTS, build_model, platform_candidates, write_pdb


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def make_integrator(args, seed: int):
    integrator = LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        args.friction_per_ps / unit.picosecond,
        args.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    return integrator


def make_simulation(topology, system, integrator, requested_platform: str):
    failures = []
    for candidate, properties in platform_candidates(requested_platform):
        try:
            simulation = Simulation(
                topology,
                system,
                integrator,
                candidate,
                properties,
            )
            if failures:
                print(
                    "Platform fallback: " + "; ".join(failures)
                    + f"; using {candidate.getName()}",
                    flush=True,
                )
            return simulation, candidate
        except Exception as exc:
            failures.append(f"{candidate.getName()}: {exc}")
            if requested_platform != "auto":
                raise
    raise RuntimeError("No OpenMM platform could create a context: " + "; ".join(failures))


def prepare_replica(args, replica_dir: Path, mutation: str | None, seed: int):
    started = time.time()
    modeller = build_model(args.pdb, mutation, args.ph)
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
    barostat = MonteCarloBarostat(
        args.pressure_bar * unit.bar,
        args.temperature_k * unit.kelvin,
        25,
    )
    barostat.setRandomNumberSeed(seed + 100_000)
    system.addForce(barostat)

    prepared_pdb = replica_dir / "prepared.pdb"
    system_xml = replica_dir / "system.xml"
    write_pdb(prepared_pdb, modeller.topology, modeller.positions)
    system_xml.write_text(XmlSerializer.serialize(system))

    integrator = make_integrator(args, seed)
    simulation, platform = make_simulation(
        modeller.topology,
        system,
        integrator,
        args.platform,
    )
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
    simulation.context.setVelocitiesToTemperature(
        args.temperature_k * unit.kelvin,
        seed,
    )
    equilibration_steps = round(
        args.equilibration_ps * 1000.0 / args.timestep_fs
    )
    simulation.step(equilibration_steps)
    equilibrated = simulation.context.getState(getPositions=True)
    write_pdb(
        replica_dir / "equilibrated.pdb",
        modeller.topology,
        equilibrated.getPositions(),
    )
    simulation.saveCheckpoint(str(replica_dir / "latest.chk"))
    setup = {
        "variant": args.variant,
        "mutation": mutation or "WT",
        "replica": args.replica,
        "source_pdb": str(args.pdb.resolve()),
        "force_field_files": forcefield_files,
        "solvent": "explicit",
        "ensemble": "NPT",
        "pH_used_for_hydrogen_placement": args.ph,
        "temperature_K": args.temperature_k,
        "pressure_bar": args.pressure_bar,
        "ionic_strength_M": args.ionic_strength_molar,
        "padding_nm": args.padding_nm,
        "cutoff_nm": args.cutoff_nm,
        "timestep_fs": args.timestep_fs,
        "friction_per_ps": args.friction_per_ps,
        "minimize_iterations": args.minimize_iterations,
        "equilibration_ps": args.equilibration_ps,
        "equilibration_steps": equilibration_steps,
        "seed": seed,
        "platform": platform.getName(),
        "atoms": modeller.topology.getNumAtoms(),
        "residues": modeller.topology.getNumResidues(),
        "setup_wall_seconds": time.time() - started,
    }
    atomic_json(replica_dir / "setup_metadata.json", setup)
    print(
        f"Prepared {args.variant} replica {args.replica}: "
        f"{setup['atoms']} atoms on {setup['platform']}",
        flush=True,
    )
    return simulation, modeller.topology, system, platform, setup


def restore_replica(args, replica_dir: Path, seed: int):
    setup = json.loads((replica_dir / "setup_metadata.json").read_text())
    pdb = PDBFile(str(replica_dir / "equilibrated.pdb"))
    system = XmlSerializer.deserialize((replica_dir / "system.xml").read_text())
    integrator = make_integrator(args, seed)
    simulation, platform = make_simulation(
        pdb.topology,
        system,
        integrator,
        args.platform,
    )
    simulation.loadCheckpoint(str(replica_dir / "latest.chk"))
    return simulation, pdb.topology, system, platform, setup


def production_ps(simulation, setup: dict, timestep_fs: float) -> float:
    production_steps = max(
        0,
        int(simulation.currentStep) - int(setup["equilibration_steps"]),
    )
    return production_steps * timestep_fs / 1000.0


def run_segment(args) -> dict:
    if args.variant not in VARIANTS:
        raise ValueError(f"Unknown variant {args.variant!r}")
    if args.replica < 0:
        raise ValueError("--replica must be non-negative")
    if args.target_ns <= 0 or args.segment_ns <= 0:
        raise ValueError("--target-ns and --segment-ns must be positive")

    replica_dir = args.output_dir / args.variant / f"replica_{args.replica}"
    replica_dir.mkdir(parents=True, exist_ok=True)
    mutation = VARIANTS[args.variant]
    seed = args.base_seed + args.replica
    setup_path = replica_dir / "setup_metadata.json"

    if setup_path.exists():
        simulation, topology, system, platform, setup = restore_replica(
            args,
            replica_dir,
            seed,
        )
    else:
        simulation, topology, system, platform, setup = prepare_replica(
            args,
            replica_dir,
            mutation,
            seed,
        )

    target_ps = args.target_ns * 1000.0
    segment_ps = args.segment_ns * 1000.0
    current_ps = production_ps(simulation, setup, args.timestep_fs)
    if current_ps >= target_ps - 1e-6:
        steps_to_run = 0
        boundary_ps = target_ps
    else:
        completed_segments = math.floor((current_ps + 1e-6) / segment_ps)
        boundary_ps = min(target_ps, (completed_segments + 1) * segment_ps)
        steps_to_run = round(
            (boundary_ps - current_ps) * 1000.0 / args.timestep_fs
        )

    trajectory_path = replica_dir / "trajectory.xtc"
    state_path = replica_dir / "state.csv"
    report_steps = max(
        1,
        round(args.report_interval_ps * 1000.0 / args.timestep_fs),
    )
    checkpoint_steps = max(
        1,
        round(args.checkpoint_interval_ps * 1000.0 / args.timestep_fs),
    )
    simulation.reporters.append(
        XTCReporter(
            str(trajectory_path),
            report_steps,
            append=trajectory_path.exists(),
        )
    )
    simulation.reporters.append(
        StateDataReporter(
            str(state_path),
            report_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            separator=",",
            append=state_path.exists(),
        )
    )
    simulation.reporters.append(
        CheckpointReporter(
            str(replica_dir / "latest.chk"),
            checkpoint_steps,
        )
    )
    if steps_to_run:
        simulation.reporters.append(
            StateDataReporter(
                os.sys.stdout,
                max(report_steps, steps_to_run // 10),
                step=True,
                time=True,
                temperature=True,
                speed=True,
                separator="\t",
            )
        )

    started = time.time()
    if steps_to_run:
        print(
            f"Advancing {args.variant} replica {args.replica} from "
            f"{current_ps / 1000.0:.6f} ns to {boundary_ps / 1000.0:.6f} ns",
            flush=True,
        )
        simulation.step(steps_to_run)
    simulation.saveCheckpoint(str(replica_dir / "latest.chk"))
    final_state = simulation.context.getState(
        getPositions=True,
        getEnergy=True,
    )
    current_ps = production_ps(simulation, setup, args.timestep_fs)
    write_pdb(replica_dir / "latest.pdb", topology, final_state.getPositions())

    campaign_state_path = replica_dir / "campaign_state.json"
    previous = {}
    if campaign_state_path.exists():
        previous = json.loads(campaign_state_path.read_text())
    history = list(previous.get("segments", []))
    history.append({
        "completed_production_ps": current_ps,
        "target_boundary_ps": boundary_ps,
        "steps_run": steps_to_run,
        "wall_seconds": time.time() - started,
        "checkpoint_step": int(simulation.currentStep),
    })
    completed = current_ps >= target_ps - args.timestep_fs / 1000.0
    campaign_state = {
        "variant": args.variant,
        "mutation": mutation or "WT",
        "replica": args.replica,
        "completed_production_ps": current_ps,
        "target_production_ps": target_ps,
        "segment_ps": segment_ps,
        "completed": completed,
        "latest_checkpoint": str((replica_dir / "latest.chk").resolve()),
        "segments": history,
    }
    atomic_json(campaign_state_path, campaign_state)

    if completed:
        metadata = {
            **setup,
            "target_production_ns": args.target_ns,
            "production_ps": current_ps,
            "report_interval_ps": args.report_interval_ps,
            "checkpoint_interval_ps": args.checkpoint_interval_ps,
            "trajectory": str(trajectory_path.resolve()),
            "trajectory_frames_expected_without_restarts": int(
                round(target_ps / args.report_interval_ps)
            ),
            "segments_completed": len(history),
            "final_potential_energy_kj_mol": (
                final_state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
            ),
        }
        atomic_json(replica_dir / "metadata.json", metadata)
        write_pdb(replica_dir / "final.pdb", topology, final_state.getPositions())
        print(
            f"Completed {args.variant} replica {args.replica}: "
            f"{current_ps / 1000.0:.6f} ns",
            flush=True,
        )
    else:
        print(
            f"Checkpointed {args.variant} replica {args.replica}: "
            f"{current_ps / 1000.0:.6f}/{args.target_ns:.6f} ns",
            flush=True,
        )
    return campaign_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--pdb", type=Path, default=Path(__file__).with_name("2LZM.pdb"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("campaign_runs"),
    )
    parser.add_argument("--target-ns", type=float, default=100.0)
    parser.add_argument("--segment-ns", type=float, default=50.0)
    parser.add_argument("--equilibration-ps", type=float, default=100.0)
    parser.add_argument("--report-interval-ps", type=float, default=10.0)
    parser.add_argument("--checkpoint-interval-ps", type=float, default=100.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--ph", type=float, default=5.4)
    parser.add_argument("--ionic-strength-molar", type=float, default=0.15)
    parser.add_argument("--padding-nm", type=float, default=1.0)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--minimize-iterations", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=20260724)
    parser.add_argument(
        "--platform",
        choices=["auto", "CUDA", "OpenCL", "CPU", "Reference"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    state = run_segment(parse_args())
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
