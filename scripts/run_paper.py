"""Reproduce selected final same-report conditions with the published settings."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from mecc.data import synthetic
from mecc.experiment import run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path("configs/final.json"))
    parser.add_argument("--scenario",nargs="+",help="Scenario keys; default: all six inputs")
    parser.add_argument("--data-dir",type=Path,default=Path("data"))
    parser.add_argument("--output",type=Path,default=Path("outputs/paper"))
    parser.add_argument("--repetitions",type=int,help="Use the first N configured report seeds")
    parser.add_argument("--epsilon",type=float,nargs="+",help="Select configured privacy budgets")
    parser.add_argument("--full-mle",action="store_true",help="Include the full-report likelihood baseline")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    scenarios=[s for s in config["scenarios"] if args.scenario is None or s["name"] in args.scenario]
    if not scenarios or (args.scenario and set(args.scenario)!={s["name"] for s in scenarios}):
        parser.error("Unknown scenario")
    if args.repetitions is not None and not 1<=args.repetitions<=len(config["report_seeds"]):
        parser.error("Repetitions must lie between 1 and the configured count")
    eps=config["epsilons"] if args.epsilon is None else args.epsilon
    if any(e not in config["epsilons"] for e in eps):parser.error("Unknown privacy budget")
    seeds=config["report_seeds"][:args.repetitions]
    decoder={k:config[k] for k in ["train_fraction","max_controls","split_seed","tie_seed","tail_tolerance"]}
    mle={"max_iter":config["mle_max_iter"],"tol":config["mle_tol"],
         "max_coordinate_steps":config["mle_max_coordinate_steps"]}
    for case in scenarios:
        if "distribution" in case:
            values=synthetic(case["n"],case["d"],case["distribution"],case["input_seed"],case["exponent"])
        else:
            path=args.data_dir/case["values_file"]
            if not path.is_file():
                raise FileNotFoundError(f"Prepare {case['values_file']} using the README data commands")
            values=np.load(path,allow_pickle=False)
        buffer=io.BytesIO();np.save(buffer,values,allow_pickle=False)
        if hashlib.sha256(buffer.getvalue()).hexdigest()!=case["values_npy_sha256"]:
            raise ValueError(f"Input fingerprint differs for {case['name']}; check extraction and sampling")
        run(values,case["d"],eps,seeds,args.output/case["name"],include_full_mle=args.full_mle,
            decoder_options=decoder,mle_options=mle)
        print(json.dumps({"scenario":case["name"],"conditions":len(eps),"repetitions":len(seeds)}),flush=True)


if __name__=="__main__":main()
