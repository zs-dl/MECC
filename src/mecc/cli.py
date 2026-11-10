"""Command-line entry points for fixed-input evaluation and data preparation."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import synthetic, movielens_first, kosarak_clicks
from .experiment import run


def main():
    parser=argparse.ArgumentParser(prog="mecc")
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("run",help="Run all estimators on the same reports")
    p.add_argument("--values",type=Path,help="One-dimensional zero-based integer .npy array")
    p.add_argument("--distribution",choices=["zipf","uniform"],default="zipf")
    p.add_argument("--n",type=int,default=256)
    p.add_argument("--d",type=int,default=32)
    p.add_argument("--exponent",type=float,default=1.2)
    p.add_argument("--input-seed",type=int,default=7)
    p.add_argument("--epsilon",type=float,nargs="+",default=[3.0])
    p.add_argument("--report-seed",type=int,default=0)
    p.add_argument("--repetitions",type=int,default=3)
    p.add_argument("--train-fraction",type=float,default=.2)
    p.add_argument("--max-controls",type=int,default=32)
    p.add_argument("--split-seed",type=int,default=20261109)
    p.add_argument("--tie-seed",type=int,default=20261110)
    p.add_argument("--tail-tolerance",type=float,default=1e-14)
    p.add_argument("--full-mle",action="store_true")
    p.add_argument("--mle-max-iter",type=int,default=200)
    p.add_argument("--mle-tol",type=float,default=1e-7)
    p.add_argument("--mle-max-coordinate-steps",type=int,default=5000)
    p.add_argument("--save-estimates",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    p=commands.add_parser("prepare",help="Convert locally supplied public datasets")
    p.add_argument("--dataset",choices=["movielens","kosarak"],required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--n",type=int,default=1024,help="Kosarak click sample size")
    p.add_argument("--input-seed",type=int,default=1069001,help="Kosarak sampling seed")
    p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        values=movielens_first(args.source) if args.dataset=="movielens" else kosarak_clicks(args.source,args.n,args.input_seed)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("wb") as stream:np.save(stream,values,allow_pickle=False)
        print(json.dumps({"samples":len(values),"dataset":args.dataset}))
        return
    if args.repetitions<1:parser.error("--repetitions must be positive")
    values=np.load(args.values,allow_pickle=False) if args.values else synthetic(args.n,args.d,args.distribution,args.input_seed,args.exponent)
    options={k:getattr(args,k) for k in ["train_fraction","max_controls","split_seed","tie_seed","tail_tolerance"]}
    summary=run(values,args.d,args.epsilon,range(args.report_seed,args.report_seed+args.repetitions),args.output,
        include_full_mle=args.full_mle,decoder_options=options,
        mle_options={"max_iter":args.mle_max_iter,"tol":args.mle_tol,"max_coordinate_steps":args.mle_max_coordinate_steps},
        save_estimates=args.save_estimates)
    print(json.dumps({"conditions":len(args.epsilon),"repetitions":args.repetitions,"summary_rows":len(summary)}))
