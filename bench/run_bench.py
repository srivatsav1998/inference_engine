"""Benchmark orchestrator: kernels + memory bandwidth + engine E2E + PyTorch baseline.

Writes a self-describing JSON report (hardware, commit, build flags, raw results, derived
metrics) to bench/results/ and prints a markdown summary suitable for the README.

Usage (normally via `make bench` / `make bench_quick`, which build the release library first):
    python bench/run_bench.py [--quick] [--no-torch] [--no-kernels]
"""

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_EXT = "dylib" if sys.platform == "darwin" else "so"


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sysctl(key):
    return sh(["sysctl", "-n", key]) if sys.platform == "darwin" else None


def hardware_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    if sys.platform == "darwin":
        to_int = lambda v: int(v) if v and v.isdigit() else None
        info.update({
            "p_cores": to_int(sysctl("hw.perflevel0.physicalcpu")),
            "e_cores": to_int(sysctl("hw.perflevel1.physicalcpu")),
            "l1d_bytes": to_int(sysctl("hw.l1dcachesize")),
            "p_cluster_l2_bytes": to_int(sysctl("hw.perflevel0.l2cachesize")),
            "e_cluster_l2_bytes": to_int(sysctl("hw.perflevel1.l2cachesize")),
            "memory_bytes": to_int(sysctl("hw.memsize")),
            "isa_features": {
                f: sysctl(f"hw.optional.arm.{f}") == "1"
                for f in ["FEAT_DotProd", "FEAT_FP16", "FEAT_BF16", "FEAT_I8MM"]
            },
        })
    return info


def build_info():
    return {
        "commit": sh(["git", "rev-parse", "--short", "HEAD"]),
        "branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(sh(["git", "status", "--porcelain", "--untracked-files=no"])),
        "compiler": (sh(["g++", "--version"]) or "").splitlines()[0] if sh(["g++", "--version"]) else None,
        "build": "release: -std=c++20 -O3 -DNDEBUG",
    }


def run_e2e(backend, args):
    cmd = [sys.executable, os.path.join(REPO, "bench", "e2e_bench.py"), "--backend", backend,
           "--lib", args.lib, "--weights-dir", args.weights_dir,
           "--decode-tokens", str(args.decode_tokens), "--repeats", str(args.repeats),
           "--prompt-lens", *map(str, args.prompt_lens)]
    print(f"[bench] E2E {backend} ...", flush=True)
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"E2E benchmark for {backend} failed (exit {out.returncode})")
    return json.loads(out.stdout.strip().splitlines()[-1])


def run_kernels(args):
    print("[bench] kernel microbenchmarks ...", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    cmd = [args.kernel_bin, "--out", path] + (["--quick"] if args.quick else [])
    subprocess.run(cmd, cwd=REPO, check=True)
    with open(path) as f:
        data = json.load(f)
    os.unlink(path)
    return data


def agreement(a, b):
    """Length of the common greedy prefix and fraction of positions that match."""
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    n = min(len(a), len(b))
    same = sum(1 for x, y in zip(a, b) if x == y)
    return prefix, (same / n if n else 0.0)


def derive(report):
    weights_bytes = report["model"]["weights_bytes"]
    peak_bw = None
    if report.get("kernels"):
        peak_bw = max(r["gbps"] for r in report["kernels"]["read_bandwidth"])
    report["derived"] = {
        "bytes_streamed_per_decode_token": weights_bytes,
        "measured_peak_read_gbps": peak_bw,
        "decode_floor_ms_per_tok": (weights_bytes / (peak_bw * 1e9) * 1e3) if peak_bw else None,
    }
    for backend in ("engine", "torch"):
        res = report.get(backend)
        if not res:
            continue
        for r in res["results"]:
            gbps = weights_bytes / (r["decode_ms_per_tok"] / 1e3) / 1e9
            r["effective_decode_gbps"] = gbps
            r["fraction_of_peak_bw"] = gbps / peak_bw if peak_bw else None

    if report.get("engine") and report.get("torch"):
        checks = []
        for re_, rt in zip(report["engine"]["results"], report["torch"]["results"]):
            prefix, frac = agreement(re_["output_tokens"], rt["output_tokens"])
            checks.append({"prompt_len": re_["prompt_len"], "greedy_prefix_match": prefix,
                           "tokens_compared": min(len(re_["output_tokens"]), len(rt["output_tokens"])),
                           "positionwise_agreement": frac})
        report["derived"]["greedy_agreement_vs_torch"] = checks


def markdown(report):
    lines = []
    hw, b = report["hardware"], report["build"]
    lines.append(f"**Hardware:** {hw['cpu']} ({hw.get('p_cores')}P + {hw.get('e_cores')}E cores, "
                 f"{(hw.get('memory_bytes') or 0) / 2**30:.0f} GB) · **Commit:** `{b['commit']}`"
                 f"{' (dirty)' if b['dirty'] else ''} · **Build:** {b['build']}")
    d = report["derived"]
    if d.get("measured_peak_read_gbps"):
        lines.append(f"\n**Decode roofline:** {d['bytes_streamed_per_decode_token'] / 1e6:.0f} MB of fp32 weights "
                     f"streamed per token ÷ {d['measured_peak_read_gbps']:.1f} GB/s measured read bandwidth "
                     f"→ floor ≈ **{d['decode_floor_ms_per_tok']:.1f} ms/token** "
                     f"(≈ {1e3 / d['decode_floor_ms_per_tok']:.0f} tok/s).")

    lines.append("\n| Backend | Prompt | TTFT (ms) | Prefill tok/s | Decode ms/tok | Decode tok/s | % of BW floor | Avg busy cores |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for backend, label in (("engine", "Engine"), ("torch", "PyTorch (HF)")):
        res = report.get(backend)
        if not res:
            continue
        for r in res["results"]:
            frac = r.get("fraction_of_peak_bw")
            lines.append(f"| {label} | {r['prompt_len']} | {r['ttft_s'] * 1e3:.0f} | {r['prefill_tok_s']:.0f} | "
                         f"{r['decode_ms_per_tok']:.1f} | {r['decode_tok_s']:.1f} | "
                         f"{frac * 100:.0f}% | {r['cpu_util_cores']:.1f} |" if frac is not None else
                         f"| {label} | {r['prompt_len']} | {r['ttft_s'] * 1e3:.0f} | {r['prefill_tok_s']:.0f} | "
                         f"{r['decode_ms_per_tok']:.1f} | {r['decode_tok_s']:.1f} | – | {r['cpu_util_cores']:.1f} |")
    for backend, label in (("engine", "Engine"), ("torch", "PyTorch")):
        if report.get(backend):
            r = report[backend]
            lines.append(f"\n{label}: first (cold) call {r['first_call_s']:.2f}s, peak RSS {r['peak_rss_mb']:.0f} MB.")

    if d.get("greedy_agreement_vs_torch"):
        parts = [f"L={c['prompt_len']}: {c['greedy_prefix_match']}/{c['tokens_compared']}"
                 for c in d["greedy_agreement_vs_torch"]]
        lines.append(f"\n**Greedy agreement with PyTorch** (matching generated-token prefix): {', '.join(parts)}.")

    k = report.get("kernels")
    if k:
        lines.append("\n| MatMul | M×K×N | Engine ms | Engine GFLOP/s | Accelerate ms | Accelerate GFLOP/s | Engine / Accelerate |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for r in k["matmul"]:
            ratio = (r["engine_ms"] / r["accelerate_ms"]) if r.get("accelerate_ms") else None
            acc_ms = f"{r['accelerate_ms']:.3f}" if r.get("accelerate_ms") else "–"
            acc_gf = f"{r['accelerate_gflops']:.1f}" if r.get("accelerate_gflops") else "–"
            lines.append(f"| {r['op']} | {r['M']}×{r['K']}×{r['N']} | {r['engine_ms']:.3f} | {r['engine_gflops']:.1f} | "
                         f"{acc_ms} | {acc_gf} | {f'{ratio:.1f}× slower' if ratio else '–'} |")
        bw = ", ".join(f"{x['threads']} thr: {x['gbps']:.1f} GB/s" for x in k["read_bandwidth"])
        lines.append(f"\nMeasured read bandwidth — {bw}.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer prompt lengths/repeats; skips PyTorch")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--no-kernels", action="store_true")
    ap.add_argument("--lib", default=os.path.join(REPO, "bin", "release", f"libengine.{LIB_EXT}"))
    ap.add_argument("--kernel-bin", default=os.path.join(REPO, "bin", "release", "kernel_bench"))
    ap.add_argument("--weights-dir", default=os.path.join(REPO, "bin"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "bench", "results"))
    args = ap.parse_args()

    # Prompt lengths are capped at 128: the engine's fixed 90 MB activation arena overflows
    # during prefill at ~192 tokens (see README "Known issues").
    if args.quick:
        args.prompt_lens, args.decode_tokens, args.repeats, args.no_torch = [8, 128], 16, 1, True
    else:
        args.prompt_lens, args.decode_tokens, args.repeats = [8, 32, 128], 64, 3

    weights = os.path.join(args.weights_dir, "model_weights.bin")
    if not os.path.exists(weights):
        raise SystemExit(f"Real GPT-2 weights not found at {weights}; run `make weights` first.")
    if not os.path.exists(args.lib):
        raise SystemExit(f"Release library not found at {args.lib}; run `make bench_build` first.")

    report = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware_info(),
        "build": build_info(),
        "model": {"name": "gpt2 (124M)", "dtype": "float32", "weights_bytes": os.path.getsize(weights)},
        "config": {"prompt_lens": args.prompt_lens, "decode_tokens": args.decode_tokens,
                   "repeats": args.repeats, "quick": args.quick},
    }
    if not args.no_kernels:
        report["kernels"] = run_kernels(args)
    report["engine"] = run_e2e("engine", args)
    if not args.no_torch:
        report["torch"] = run_e2e("torch", args)
    derive(report)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "-quick" if args.quick else ""
    path = os.path.join(args.out_dir, f"{stamp}-{report['build']['commit']}{suffix}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + markdown(report))
    print(f"\n[bench] report written to {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
