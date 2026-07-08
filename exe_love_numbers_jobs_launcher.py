"""
1. Locally, using multiple CPU workers:

       python exe_love_numbers_jobs_launcher.py local --workers 4

2. On a Slurm cluster, using a job array:

       python exe_love_numbers_jobs_launcher.py submit --max_running 20

In the parameter_lines file, each line is one run, for example:

       {"\\alpha^{MANTLE_0}": 0.2, "Delta^{MANTLE_0}": 2.1}
       {"\\alpha^{MANTLE_0}": 0.3, "Delta^{MANTLE_0}": 2.1}
"""

from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from json import JSONDecodeError, loads
from os import cpu_count
from pathlib import Path
from shlex import quote
from subprocess import CalledProcessError, run
from sys import executable

from base_models import DEFAULT_WORKDIR

from alna import DEFAULT_PARAMETER_LINES_PATH, ROOT_PATH
from exe_love_numbers_computing import parse_general_args

DEFAULT_CLUSTER_VENV = ROOT_PATH.parent.joinpath("alna_venv")
DEFAULT_CLUSTER_PYTHON_MODULE = "python/3.11.10"
DEFAULT_SINGLE_JOB_SCRIPT = str(ROOT_PATH.joinpath("exe_love_numbers_computing.py").resolve())
LAUNCHER_PATH = Path(__file__).resolve()


def read_parameters(args: Namespace) -> list[dict[str, float]]:
    """
    Reads parameter_lines (.JSONL) file.
    """

    parameter_lines = Path(args.parameter_lines_path).joinpath(args.parameter_lines)

    if not parameter_lines.exists():

        raise FileNotFoundError(f"Parameter file does not exist: {parameter_lines}")

    parameters: list[dict[str, float]] = []

    with parameter_lines.open(encoding="utf-8") as f:

        for i, line in enumerate(f, start=1):

            line = line.strip()

            if not line:

                continue

            try:

                item = loads(line)

            except JSONDecodeError as exc:

                raise ValueError(f"Invalid JSON on line {i} of {parameter_lines}") from exc

            if not isinstance(item, dict):

                raise ValueError(f"Line {i} of {parameter_lines} is not a JSON object")

            clean_item: dict[str, float] = {}

            for name, value in item.items():

                if not isinstance(name, str):

                    raise ValueError(f"Parameter name on line {i} is not a string: {name!r}")

                try:

                    clean_item[name] = float(value)

                except (TypeError, ValueError) as exc:

                    raise ValueError(
                        f"Parameter value for {name!r} on line {i} is not a float: {value!r}"
                    ) from exc

            parameters.append(clean_item)

    return parameters


def append_common_cli_args(cmd: list[str], args: Namespace) -> None:
    """
    Appends common arguments to commands.
    """

    if args.path:

        path: Path = Path(args.path)
        cmd.extend(["--path", str(path.resolve())])

    if args.output_path:

        output_path: Path = Path(args.output_path)
        cmd.extend(["--output_path", str(output_path.resolve())])

    if args.period_tab_per_degree:

        cmd.extend(["--period_tab_per_degree", str(args.period_tab_per_degree)])

    if args.period_tab_per_degree_path:

        period_tab_per_degree_path: Path = Path(args.period_tab_per_degree_path)
        cmd.extend(["--period_tab_per_degree_path", str(period_tab_per_degree_path.resolve())])

    if args.force_transient:

        cmd.append("--force_transient")

    if args.force_not_transient:

        cmd.append("--force_not_transient")

    if args.force_viscous:

        cmd.append("--force_viscous")

    if args.force_not_viscous:

        cmd.append("--force_not_viscous")

    if args.compute_partials:

        cmd.append("--compute_partials")

    if args.not_compute_partials:

        cmd.append("--not_compute_partials")

    if args.not_format_name:

        cmd.append("--not_format_name")


def run_one_task(args: Namespace, task_id: int = 1) -> None:
    """
    Run exactly one parameter-set task: task_id=i reads line i of the (.JSONL) file.
    """

    all_parameters = read_parameters(args=args)
    cmd = [
        executable,
        DEFAULT_SINGLE_JOB_SCRIPT,
        "--name",
        args.name,
    ]
    append_common_cli_args(cmd=cmd, args=args)

    if len(all_parameters) > 0:

        cmd.append("--parameters")
        print(f"[task {task_id}] parameters: {all_parameters[task_id-1]}")

        for name, value in all_parameters[task_id - 1].items():

            cmd.extend([name, str(value)])

    print(f"[task {task_id}] command: " + " ".join(quote(s=x) for x in cmd), flush=True)
    run(args=cmd, cwd=DEFAULT_WORKDIR, check=True)


def local_worker(args: Namespace, task_id: int = 1) -> tuple[int, bool, str]:
    """
    Manages errors for local parallel computing.
    """

    try:

        run_one_task(args=args, task_id=task_id)

        return task_id, True, ""

    except (CalledProcessError, RuntimeError, IndexError, ValueError, FileNotFoundError) as exc:

        return task_id, False, repr(exc)


def run_local(args: Namespace) -> None:
    """
    Runs all jobs in parallel locally.
    """

    n_parameter_sets = len(read_parameters(args=args))
    n_tasks = max(1, n_parameter_sets)
    workers = min(args.workers, n_tasks)
    failures: list[int] = []
    print(f"Running {n_tasks} tasks locally with {workers} worker(s).")

    with ProcessPoolExecutor(max_workers=workers) as pool:

        if n_tasks == 0:

            futures = {local_worker(args=args): 0}

        else:

            futures = {
                pool.submit(local_worker, args, task_id): task_id
                for task_id in range(1, n_tasks + 1)
            }

        for future in as_completed(fs=futures):

            task_id, ok, message = future.result()

            if ok:

                print(f"[task {task_id}] finished successfully.", flush=True)

            else:

                print(f"[task {task_id}] failed: {message}", flush=True)
                failures.append(task_id)

    if failures:

        raise SystemExit(f"Some local tasks failed: {failures}")

    print("All local tasks finished successfully.")


def shell_join_multiline(cmd: list[str]) -> str:
    """
    Appends safely text lines for a Slurm command.
    """

    return " \\\n    ".join(quote_slurm_arg(x=x) for x in cmd)


def quote_slurm_arg(x: str) -> str:
    """
    Treats the task ID argument separately.
    """

    if x == "${SLURM_ARRAY_TASK_ID}":

        return '"${SLURM_ARRAY_TASK_ID}"'

    return quote(str(x))


def make_slurm_script(args: Namespace, workdir: Path = DEFAULT_WORKDIR) -> Path:
    """
    Generates an sbatch script that runs one array task.
    Each Slurm array task calls this same script in "worker" mode.
    """

    slurm_file = Path(args.slurm_file).resolve()
    slurm_file.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = workdir.joinpath("logs").resolve()
    parameter_lines_path: Path = Path(args.parameter_lines_path)
    cluster_python = Path(args.venv).resolve() / "bin" / "python"

    preamble = f"""#!/bin/bash

#SBATCH --job-name={args.job_name}
#SBATCH --time={args.walltime}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --output={logs_dir}/slurm_%A_%a.out
#SBATCH --error={logs_dir}/slurm_%A_%a.err

set -euo pipefail

echo "Job started on: $(hostname)"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-unset}}"
echo "SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID:-unset}}"

if [ -z "${{SLURM_ARRAY_TASK_ID:-}}" ]; then
    echo "This script must be submitted as a Slurm job array."
    echo "Expected SLURM_ARRAY_TASK_ID to be set."
    exit 1
fi

mkdir -p {quote(str(logs_dir.resolve()))}
cd {quote(str(workdir.resolve()))}

if command -v module >/dev/null 2>&1; then
    module purge
    module load {quote(args.python_module)}
fi

source {quote(str(Path(args.venv) / "bin" / "activate"))}
"""

    worker_cmd = [
        str(cluster_python),
        str(LAUNCHER_PATH),
        "worker",
        "--name",
        str(args.name),
        "--task_id",
        "${SLURM_ARRAY_TASK_ID}",
        "--parameter_lines",
        str(args.parameter_lines),
        "--parameter_lines_path",
        str(parameter_lines_path.resolve()),
    ]

    append_common_cli_args(cmd=worker_cmd, args=args)
    command_text = shell_join_multiline(worker_cmd)
    script = f"""{preamble}

{command_text}

echo "Job finished."
"""

    slurm_file.write_text(script, encoding="utf-8")
    return slurm_file


def submit_slurm(args: Namespace, workdir: Path = DEFAULT_WORKDIR) -> None:
    """
    Submits all jobs on the cluster.
    """

    # Make the logs directory before submitting because Slurm may expect it.
    logs_dir = workdir.joinpath("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    slurm_file = make_slurm_script(args=args, workdir=workdir)
    n_tasks = max(1, len(read_parameters(args=args)))
    array_spec = f"1-{n_tasks}"

    if args.max_running is not None:

        array_spec += f"%{args.max_running}"

    cmd = ["sbatch", f"--array={array_spec}", str(slurm_file.resolve())]
    print("Submitting Slurm job array:")
    print(" ".join(quote(x) for x in cmd))

    if args.dry_run:

        print("Dry run only. Not submitting.")
        print(f"Generated Slurm script: {slurm_file}")

        return

    try:

        result = run(cmd, text=True, capture_output=True, check=True)

    except CalledProcessError as exc:

        print("sbatch failed.")
        print(f"Generated Slurm script: {slurm_file}")
        print(f"Command: {' '.join(quote(str(x)) for x in cmd)}")

        if exc.stdout:

            print("sbatch stdout:")
            print(exc.stdout.rstrip())

        if exc.stderr:

            print("sbatch stderr:")
            print(exc.stderr.rstrip())

        raise

    print("sbatch output:")
    print(result.stdout.strip())
    print(f"Generated Slurm script: {slurm_file}")


def detect_scheduler(_: Namespace) -> None:
    """
    Print useful diagnostics for the cluster.
    """

    for command in ["qsub", "sbatch", "squeue", "scontrol"]:

        path = run(
            ["bash", "-lc", f"command -v {quote(command)} || true"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        print(f"{command}: {path or 'not found'}")

    print()
    print("Shell type information:")
    run(["bash", "-lc", "type -a qsub 2>/dev/null || true"], check=False)
    run(["bash", "-lc", "type -a sbatch 2>/dev/null || true"], check=False)
    print()


def add_common_mode_args(parser: ArgumentParser) -> None:
    """
    The parameter_lines argument is needed whether mode is local or slurm.
    """

    parser.add_argument("--parameter_lines", required=True)
    parser.add_argument(
        "--parameter_lines_path",
        help="Path to the target directory.",
        default=DEFAULT_PARAMETER_LINES_PATH,
    )


def parse_multi_job_args() -> Namespace:
    """
    Defines a parsing function for command-line arguments.
    """

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # For Local runs.
    local_parser = subparsers.add_parser("local", help="Run all tasks locally in parallel.")
    parse_general_args(parser=local_parser)
    add_common_mode_args(parser=local_parser)
    local_parser.add_argument("--workers", type=int, default=cpu_count() or 1)

    # For Slurm runs.
    submit_parser = subparsers.add_parser("submit", help="Submit a Slurm job array.")
    parse_general_args(parser=submit_parser)
    add_common_mode_args(parser=submit_parser)
    submit_parser.add_argument("--job_name", default="alna_array")
    submit_parser.add_argument("--slurm_file", default="run_alna_array.sbatch")
    submit_parser.add_argument("--walltime", default="04:00:00")
    submit_parser.add_argument("--mem", default="4G")
    submit_parser.add_argument("--cpus_per_task", type=int, default=1)
    submit_parser.add_argument("--max_running", type=int, default=None)
    submit_parser.add_argument("--venv", default=DEFAULT_CLUSTER_VENV.resolve())
    submit_parser.add_argument("--python_module", default=DEFAULT_CLUSTER_PYTHON_MODULE)
    submit_parser.add_argument("--dry_run", action="store_true")

    # For Slurm single run. Mandatory because used by Slurm.
    worker_parser = subparsers.add_parser("worker", help="Run exactly one task. Used by Slurm.")
    parse_general_args(parser=worker_parser)
    add_common_mode_args(parser=worker_parser)
    worker_parser.add_argument("--task_id", required=True)

    # For diagnostics. No need for other inputs.
    subparsers.add_parser("detect", help="Print scheduler diagnostics.")

    return parser.parse_args()


def main() -> None:
    """
    Manages local/cluster runs and mandatory Slurmm modes.
    """

    args = parse_multi_job_args()

    if args.mode == "local":

        run_local(args)

    elif args.mode == "submit":

        submit_slurm(args)

    elif args.mode == "worker":

        run_one_task(
            args=args,
            task_id=int(args.task_id),
        )

    elif args.mode == "detect":

        detect_scheduler(args)

    else:

        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":

    main()
