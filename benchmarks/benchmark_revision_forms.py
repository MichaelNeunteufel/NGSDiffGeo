"""Revision-neutral public-API benchmark for form performance.

Unlike ``benchmark_compressed_forms.py``, this runner deliberately uses no
private compact or dense factory.  The same file can therefore be executed
against an isolated ``main`` build and a candidate build.  Timing thresholds
belong in stored comparison reports, not CI; tests only smoke-test execution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ngsolve
import ngsdiffgeo as dg
from netgen.occ import unit_cube
from ngsolve import (
    BilinearForm,
    InnerProduct,
    Mesh,
    Norm,
    NumberSpace,
    SymbolicBFI,
    TaskManager,
)

from benchmark_common import (
    _extension_identity,
    _file_identity,
    _peak_rss_bytes,
    _repository_identity,
    _retained_memory_sample,
    _statistics,
    _subprocess_json,
    _wrapper_identity,
)
from benchmark_compressed_forms import (
    _DEFAULT_WORKLOAD_NAMES,
    _HIGH_ORDER_WORKLOAD_NAMES,
    _workloads,
)


_WORKLOAD_NAMES = _DEFAULT_WORKLOAD_NAMES + _HIGH_ORDER_WORKLOAD_NAMES
_APPLICATION_NAME = "mixed_forms_energy_3d"
_PHASES = ("construction", "compile", "form_setup", "assembly", "total")


def _coefficient(form):
    return form.coef


def _value_norm(form, mesh):
    coefficient = _coefficient(form)
    squared = (
        coefficient * coefficient
        if not coefficient.dims
        else InnerProduct(coefficient, coefficient)
    )
    return math.sqrt(max(0.0, float(ngsolve.Integrate(squared, mesh))))


def _benchmark_construction(factory, warmup, iterations):
    candidate = None
    for _ in range(warmup):
        candidate = factory()
    samples = []
    for _ in range(iterations):
        candidate = None
        started = time.perf_counter_ns()
        candidate = factory()
        samples.append((time.perf_counter_ns() - started) * 1e-9)
    return candidate, _statistics(samples)


def _benchmark_compile(factory, warmup, iterations):
    compiled = None
    for _ in range(warmup):
        compiled = _coefficient(factory()).Compile(
            realcompile=False, wait=True, maxderiv=0
        )
    samples = []
    for _ in range(iterations):
        coefficient = _coefficient(factory())
        compiled = None
        started = time.perf_counter_ns()
        compiled = coefficient.Compile(
            realcompile=False, wait=True, maxderiv=0
        )
        samples.append((time.perf_counter_ns() - started) * 1e-9)
    return compiled, {"kind": "ngsolve_graph", **_statistics(samples)}


def _rss_bytes(who):
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(who).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _generated_artifact_sizes(root):
    root = Path(root)
    files = [path for path in root.rglob("*") if path.is_file()]

    def total(predicate):
        return sum(path.stat().st_size for path in files if predicate(path))

    sources = [
        path for path in files if path.suffix in {".cpp", ".cc", ".cxx"}
    ]
    return {
        "source_bytes": sum(path.stat().st_size for path in sources),
        "source_lines": sum(
            len(path.read_bytes().splitlines()) for path in sources
        ),
        "object_bytes": total(lambda path: path.suffix in {".o", ".obj"}),
        "library_bytes": total(
            lambda path: path.suffix in {".so", ".dylib", ".dll"}
        ),
        "file_count": len(files),
    }


def _assembled_value(coefficient, mesh, *, simd):
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    squared = (
        coefficient * coefficient
        if not coefficient.dims
        else InnerProduct(coefficient, coefficient)
    )
    form = BilinearForm(space)
    integrator = SymbolicBFI(
        squared * trial * test,
        simd_evaluate=simd,
    )
    form += integrator
    with TaskManager():
        form.Assemble()
    return float(Norm(form.mat.AsVector())), bool(integrator.simd_evaluate)


def _realcompile_worker(workload_name, threads, maxh, layout):
    ngsolve.SetNumThreads(threads)
    ngsolve.ngsglobals.code_uses_tensors = layout == "tensor_temporaries"
    workload = _workloads()[workload_name]
    coefficient = _coefficient(workload["factory"]())
    graph_compiled = coefficient.Compile(
        realcompile=False, wait=True, maxderiv=0
    )
    mesh = Mesh(unit_cube.GenerateMesh(maxh=maxh))

    try:
        import resource
    except ImportError:
        resource = None
    self_before = (
        _rss_bytes(resource.RUSAGE_SELF) if resource is not None else None
    )
    child_before = (
        _rss_bytes(resource.RUSAGE_CHILDREN) if resource is not None else None
    )
    started = time.perf_counter_ns()
    compiled = coefficient.Compile(
        realcompile=True,
        wait=True,
        maxderiv=0,
        keep_files=True,
    )
    elapsed = (time.perf_counter_ns() - started) * 1e-9
    self_after = (
        _rss_bytes(resource.RUSAGE_SELF) if resource is not None else None
    )
    child_after = (
        _rss_bytes(resource.RUSAGE_CHILDREN) if resource is not None else None
    )

    values = {}
    maximum_error = 0.0
    for backend, simd in (("scalar", False), ("simd", True)):
        reference, reference_simd = _assembled_value(
            graph_compiled, mesh, simd=simd
        )
        candidate, candidate_simd = _assembled_value(
            compiled, mesh, simd=simd
        )
        error = abs(candidate - reference) / max(1.0, abs(reference))
        maximum_error = max(maximum_error, error)
        values[backend] = {
            "reference_norm": reference,
            "realcompiled_norm": candidate,
            "relative_value_error": error,
            "reference_simd_active": reference_simd,
            "realcompiled_simd_active": candidate_simd,
        }

    artifact_root = os.environ.get("NGSDIFFGEO_REALCOMPILE_ARTIFACT_ROOT")
    return {
        "status": "completed",
        "workload": workload_name,
        "layout": layout,
        "compile_seconds": elapsed,
        "compiler_cache": "disabled",
        "generated_artifacts": (
            _generated_artifact_sizes(artifact_root)
            if artifact_root
            else None
        ),
        "process_peak_rss_before_bytes": self_before,
        "process_peak_rss_after_bytes": self_after,
        "compiler_peak_rss_before_bytes": child_before,
        "compiler_peak_rss_after_bytes": child_after,
        "maximum_relative_value_error": maximum_error,
        "values": values,
        "compiled_tree": dg.CFStats(compiled),
    }


def _terminate_process_group(process):
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
            process.kill()
        process.communicate()


def _realcompile_sample(workload_name, threads, maxh, timeout, layout):
    with tempfile.TemporaryDirectory(
        prefix="ngsdiffgeo-realcompile-"
    ) as artifact_root:
        environment = os.environ.copy()
        environment["CCACHE_DISABLE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TMPDIR"] = artifact_root
        environment["NGSDIFFGEO_REALCOMPILE_ARTIFACT_ROOT"] = artifact_root
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--realcompile-worker",
                workload_name,
                "--threads",
                str(threads),
                "--maxh",
                str(maxh),
                "--realcompile-layout",
                layout,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(
                timeout=None if timeout == 0 else timeout
            )
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return {
                "status": "timed_out",
                "workload": workload_name,
                "layout": layout,
                "timeout_seconds": timeout,
                "generated_artifacts": _generated_artifact_sizes(
                    artifact_root
                ),
            }
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                output=stdout,
                stderr=stderr,
            )
        lines = [
            line for line in stdout.splitlines() if line.strip()
        ]
        if not lines:
            raise RuntimeError("realcompile worker produced no JSON output")
        return json.loads(lines[-1])


def _sample_statistic(samples, key):
    values = [sample[key] for sample in samples]
    return {
        "median": statistics.median(values),
        "samples": values,
    }


def _benchmark_realcompile(
    workload_name, warmup, iterations, threads, maxh, tolerance, timeout,
    layout,
):
    for _ in range(warmup):
        _realcompile_sample(workload_name, threads, maxh, timeout, layout)
    samples = [
        _realcompile_sample(workload_name, threads, maxh, timeout, layout)
        for _ in range(iterations)
    ]
    completed = [
        sample for sample in samples if sample.get("status") != "timed_out"
    ]
    if not completed:
        return None, {
            "kind": "native_cpp",
            "layout": layout,
            "status": "censored_timeout",
            "median_seconds": None,
            "minimum_seconds": None,
            "maximum_seconds": None,
            "samples_seconds": [],
            "timeout_seconds": timeout,
            "timed_out_samples": len(samples),
            "process_model": "one fresh Python process per sample",
            "compiler_cache": "disabled with CCACHE_DISABLE=1",
            "samples": samples,
        }
    timing = _statistics(
        [sample["compile_seconds"] for sample in completed]
    )
    timing.update(
        {
            "kind": "native_cpp",
            "layout": layout,
            "status": (
                "completed"
                if len(completed) == len(samples)
                else "partially_censored_timeout"
            ),
            "timeout_seconds": timeout or None,
            "timed_out_samples": len(samples) - len(completed),
            "process_model": "one fresh Python process per sample",
            "compiler_cache": "disabled with CCACHE_DISABLE=1",
            "generated_source_bytes": _sample_statistic(
                [sample["generated_artifacts"] for sample in completed],
                "source_bytes",
            ),
            "generated_source_lines": _sample_statistic(
                [sample["generated_artifacts"] for sample in completed],
                "source_lines",
            ),
            "compiler_peak_rss_bytes": _sample_statistic(
                completed, "compiler_peak_rss_after_bytes"
            ),
            "maximum_relative_value_error": max(
                sample["maximum_relative_value_error"]
                for sample in completed
            ),
            "samples": samples,
        }
    )
    if timing["maximum_relative_value_error"] > tolerance:
        raise RuntimeError(
            "graph/native-compile value mismatch for "
            f"{workload_name}: {timing['maximum_relative_value_error']:.3e} "
            f"exceeds {tolerance:.3e}"
        )
    return completed[-1]["compiled_tree"], timing


def _application_factory():
    workloads = _workloads()
    kform_factory = workloads["kform_wedge_d_3d"]["factory"]
    double_factory = workloads["double_form_arithmetic_22_3d"]["factory"]

    def factory():
        kform = kform_factory()
        double_form = double_factory()
        return InnerProduct(kform, kform) + 0.125 * InnerProduct(
            double_form, double_form
        )

    return factory


def _run_pipeline(factory, space, trial, test, simd):
    started = time.perf_counter_ns()
    coefficient = factory()
    constructed = time.perf_counter_ns()
    compiled = coefficient.Compile(realcompile=False, wait=True, maxderiv=0)
    compilation_finished = time.perf_counter_ns()
    form = BilinearForm(space)
    integrator = SymbolicBFI(
        compiled * trial * test,
        simd_evaluate=simd,
    )
    form += integrator
    form_finished = time.perf_counter_ns()
    form.Assemble()
    assembled = time.perf_counter_ns()
    return (
        (coefficient, compiled, form, integrator),
        {
            "construction": (constructed - started) * 1e-9,
            "compile": (compilation_finished - constructed) * 1e-9,
            "form_setup": (form_finished - compilation_finished) * 1e-9,
            "assembly": (assembled - form_finished) * 1e-9,
            "total": (assembled - started) * 1e-9,
        },
    )


def _benchmark_application(args, mesh):
    factory = _application_factory()
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    simd = args.backend == "simd"
    bundle = None
    samples = {phase: [] for phase in _PHASES}
    with TaskManager():
        for _ in range(args.warmup):
            bundle, _ = _run_pipeline(
                factory, space, trial, test, simd
            )
        for _ in range(args.iterations):
            bundle = None
            bundle, sample = _run_pipeline(
                factory, space, trial, test, simd
            )
            for phase in _PHASES:
                samples[phase].append(sample[phase])

        for _ in range(args.reuse_warmup):
            bundle[2].Assemble()
        reuse_samples = []
        for _ in range(args.reuse_iterations):
            started = time.perf_counter_ns()
            bundle[2].Assemble()
            reuse_samples.append((time.perf_counter_ns() - started) * 1e-9)

    return {
        "phases": {
            phase: _statistics(values) for phase, values in samples.items()
        },
        "reuse_assembly": _statistics(reuse_samples),
        "value_norm": float(Norm(bundle[2].mat.AsVector())),
        "coefficient_tree": dg.CFStats(bundle[0]),
        "compiled_tree": dg.CFStats(bundle[1]),
        "simd_requested": simd,
        "simd_active": bool(bundle[3].simd_evaluate),
    }


def _memory_sample(workload_name, count):
    factory = _workloads()[workload_name]["factory"]
    return _retained_memory_sample(
        factory,
        count,
        count_key="retained_graphs",
        per_item_key="estimated_incremental_bytes_per_graph",
    )


def _memory_report(workload_name, count):
    return _subprocess_json(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--memory-worker",
            workload_name,
            "--retained-graphs",
            str(count),
        ]
    )


def _application_memory_sample(args):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    space = NumberSpace(mesh)
    trial, test = space.TnT()
    factory = _application_factory()
    simd = args.backend == "simd"
    def build_pipeline():
        bundle, _ = _run_pipeline(factory, space, trial, test, simd)
        return bundle

    return _retained_memory_sample(
        build_pipeline,
        args.retained_pipelines,
        count_key="retained_pipelines",
        per_item_key="estimated_incremental_bytes_per_pipeline",
        context_factory=TaskManager,
    )


def _application_memory_report(args):
    return _subprocess_json(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--pipeline-memory-worker",
            "--retained-pipelines",
            str(args.retained_pipelines),
            "--backend",
            args.backend,
            "--maxh",
            str(args.maxh),
            "--threads",
            str(args.threads),
        ]
    )


def run(args):
    ngsolve.SetNumThreads(args.threads)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=args.maxh))
    selected = args.workload or list(_DEFAULT_WORKLOAD_NAMES)
    workloads = _workloads()
    results = {}
    if args.mode in {"construction", "compile", "all"}:
        for name in selected:
            factory = workloads[name]["factory"]
            output, construction = _benchmark_construction(
                factory, args.construction_warmup, args.construction_iterations
            )
            result = {
                "space_dimension": workloads[name]["space_dimension"],
                "form_degrees": workloads[name]["form_degrees"],
                "construction": construction,
                "wrapper_tree": dg.CFStats(output),
                "value_norm": _value_norm(output, mesh),
                "compile": None,
                "compiled_tree": None,
                "retained_graph_memory": None,
            }
            if args.mode in {"compile", "all"}:
                if args.compile_kind == "real":
                    compiled_tree, compile_statistics = _benchmark_realcompile(
                        name,
                        args.compile_warmup,
                        args.compile_iterations,
                        args.threads,
                        args.maxh,
                        args.tolerance,
                        args.realcompile_timeout,
                        args.realcompile_layout,
                    )
                else:
                    compiled, compile_statistics = _benchmark_compile(
                        factory, args.compile_warmup, args.compile_iterations
                    )
                    compiled_tree = dg.CFStats(compiled)
                result["compile"] = compile_statistics
                result["compiled_tree"] = compiled_tree
            if args.retained_graphs:
                result["retained_graph_memory"] = _memory_report(
                    name, args.retained_graphs
                )
            results[name] = result

    application = None
    if args.mode in {"application", "all"}:
        application = _benchmark_application(args, mesh)
        application["retained_pipeline_memory"] = (
            _application_memory_report(args)
            if args.retained_pipelines
            else None
        )

    return {
        "benchmark": "revision_neutral_forms",
        "label": args.label,
        "mode": args.mode,
        "compile_kind": args.compile_kind,
        "workloads": selected,
        "ngsolve_version": ngsolve.__version__,
        "ngsdiffgeo_version": getattr(dg, "__version__", "unknown"),
        "ngsdiffgeo_extension": _extension_identity(),
        "ngsdiffgeo_wrapper": _wrapper_identity(),
        "benchmark_script": _file_identity(__file__),
        "repository": _repository_identity(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "threads": args.threads,
        "backend": args.backend,
        "maxh": args.maxh,
        "elements": mesh.ne,
        "construction_warmup": args.construction_warmup,
        "construction_iterations": args.construction_iterations,
        "compile_warmup": args.compile_warmup,
        "compile_iterations": args.compile_iterations,
        "realcompile_timeout": args.realcompile_timeout or None,
        "realcompile_layout": args.realcompile_layout,
        "tolerance": args.tolerance,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "reuse_warmup": args.reuse_warmup,
        "reuse_iterations": args.reuse_iterations,
        "retained_graphs": args.retained_graphs,
        "retained_pipelines": args.retained_pipelines,
        "measurement_boundaries": {
            "construction": "fresh public-API expression root",
            "compile": (
                "Compile(realcompile=True, wait=True, maxderiv=0); factory "
                "excluded; fresh process and CCACHE_DISABLE=1 per sample"
                if args.compile_kind == "real"
                else "Compile(realcompile=False); factory excluded"
            ),
            "application": "public construction through first assembly",
            "reuse_assembly": "Assemble on one retained prepared pipeline",
            "memory": (
                "fresh subprocess peak-RSS increase after one warmed graph"
            ),
        },
        "results": results,
        "application": {
            "name": _APPLICATION_NAME,
            "result": application,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument(
        "--workload", action="append", choices=_WORKLOAD_NAMES
    )
    parser.add_argument(
        "--mode",
        choices=("construction", "compile", "application", "all"),
        default="all",
    )
    parser.add_argument(
        "--compile-kind", choices=("graph", "real"), default="graph"
    )
    parser.add_argument("--backend", choices=("scalar", "simd"), default="simd")
    parser.add_argument("--maxh", type=float, default=0.3)
    parser.add_argument("--construction-warmup", type=int, default=10)
    parser.add_argument("--construction-iterations", type=int, default=100)
    parser.add_argument("--compile-warmup", type=int, default=2)
    parser.add_argument("--compile-iterations", type=int, default=5)
    parser.add_argument(
        "--realcompile-timeout",
        type=float,
        default=0,
        help="seconds per native-compile sample; zero disables the timeout",
    )
    parser.add_argument(
        "--realcompile-layout",
        choices=("scalar_variables", "tensor_temporaries"),
        default="scalar_variables",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--reuse-warmup", type=int, default=3)
    parser.add_argument("--reuse-iterations", type=int, default=31)
    parser.add_argument("--retained-graphs", type=int, default=0)
    parser.add_argument("--retained-pipelines", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--memory-worker", choices=_WORKLOAD_NAMES, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--pipeline-memory-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--realcompile-worker", choices=_WORKLOAD_NAMES, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.maxh <= 0:
        parser.error("--maxh must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    for name in (
        "construction_warmup",
        "compile_warmup",
        "warmup",
        "reuse_warmup",
        "retained_graphs",
        "retained_pipelines",
        "realcompile_timeout",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    for name in (
        "construction_iterations",
        "compile_iterations",
        "iterations",
        "reuse_iterations",
        "threads",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.memory_worker and args.retained_graphs < 1:
        parser.error("memory worker requires --retained-graphs > 0")
    if args.pipeline_memory_worker and args.retained_pipelines < 1:
        parser.error("pipeline memory worker requires --retained-pipelines > 0")
    return args


def main():
    args = parse_args()
    if args.realcompile_worker:
        print(
            json.dumps(
                _realcompile_worker(
                    args.realcompile_worker,
                    args.threads,
                    args.maxh,
                    args.realcompile_layout,
                ),
                sort_keys=True,
            )
        )
        return
    if args.memory_worker:
        print(
            json.dumps(
                _memory_sample(args.memory_worker, args.retained_graphs),
                sort_keys=True,
            )
        )
        return
    if args.pipeline_memory_worker:
        print(json.dumps(_application_memory_sample(args), sort_keys=True))
        return
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(rendered)


if __name__ == "__main__":
    main()
