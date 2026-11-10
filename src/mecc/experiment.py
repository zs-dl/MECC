"""Same-report evaluation on fixed empirical inputs."""
import csv
import json
from pathlib import Path
import numpy as np
from .baselines import calibrate_pl0
from .decoder import decode
from .full_mle import estimate as full_mle
from .numerics import simplex_projection
from .reports import privatize, column_counts, validate_values


def errors(values, estimate, d):
    truth = np.bincount(values, minlength=d)/len(values)
    error = np.asarray(estimate)-truth
    return {"mse": float(np.mean(error**2)), "l1": float(np.abs(error).sum()),
            "linf": float(np.abs(error).max())}


def estimates_from_reports(packed, d, epsilon, *, include_full_mle=False,
                           decoder_options=None, mle_options=None):
    n = len(packed)
    estimates, detail, state = decode(packed, n, d, epsilon, **(decoder_options or {}))
    counts = column_counts(packed, d)
    q = 1/(np.exp(epsilon)+1)
    raw = (counts/n-q)/(0.5-q)
    output = {"mecc": estimates["mecc"], "mecc_pre": estimates["mecc_pre"],
              "calibrate_pl0_public": estimates["calibrate_public"],
              "calibrate_pl0_public_pre": estimates["calibrate_pre"],
              "oue_raw": raw, "oue_simplex": simplex_projection(raw)}
    fitted = calibrate_pl0(counts, n, epsilon, simplex_projection, include_mle=True)
    output["calibrate_pl0_mle"] = fitted["calibrate_pl0_mle"][0]
    diagnostics = {"mecc": detail}
    if include_full_mle:
        output["oue_full_mle"], diagnostics["oue_full_mle"] = full_mle(
            packed, n, d, epsilon, **(mle_options or {}))
        if not diagnostics["oue_full_mle"]["success"]:
            raise RuntimeError("Full-report MLE did not meet its full-domain gap tolerance")
    return output, diagnostics, state


def run(values, d, epsilons, seeds, output, *, include_full_mle=False,
        decoder_options=None, mle_options=None, save_estimates=False):
    values = validate_values(values, d)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostics = []
    epsilons, seeds = list(epsilons), list(seeds)
    if not epsilons or not seeds:
        raise ValueError("At least one privacy budget and report seed are required")
    for epsilon in epsilons:
        for seed in seeds:
            packed = privatize(values, d, float(epsilon), seed=int(seed))
            estimates, detail, _ = estimates_from_reports(packed, d, float(epsilon),
                include_full_mle=include_full_mle, decoder_options=decoder_options, mle_options=mle_options)
            for method, estimate in estimates.items():
                rows.append({"epsilon": float(epsilon), "seed": int(seed), "method": method,
                             **errors(values, estimate, d)})
            diagnostics.append({"epsilon": float(epsilon), "seed": int(seed),
                                "mecc_numerical_valid": detail["mecc"]["numerical_valid"],
                                "mle_gap": detail.get("oue_full_mle", {}).get("duality_gap")})
            if save_estimates:
                np.savez_compressed(output/f"estimates-eps{epsilon:g}-seed{seed}.npz", **estimates)
    with (output/"metrics.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["epsilon"], row["method"]), []).append(row)
    summary = []
    for (epsilon,method), group in groups.items():
        row={"epsilon":epsilon,"method":method,"repetitions":len(group)}
        for metric in ("mse","l1","linf"):
            vals=np.array([r[metric] for r in group])
            row[metric+"_mean"]=float(vals.mean())
            row[metric+"_std"]=float(vals.std(ddof=1)) if len(vals)>1 else None
        summary.append(row)
    with (output/"summary.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    settings={"n":len(values),"d":d,"epsilons":epsilons,"report_seeds":seeds,
              "target":"fixed empirical frequency", "report_generator":"PCG64",
              "decoder_options":decoder_options or {},"include_full_mle":include_full_mle,
              "mle_options":mle_options or {},"diagnostics":diagnostics}
    (output/"settings.json").write_text(json.dumps(settings,indent=2,allow_nan=False)+"\n")
    return summary
