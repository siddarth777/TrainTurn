import _bootstrap
import argparse
import json
from railway_rl.config import load_config
from railway_rl.evaluation.evaluator import evaluate


def main():
    p=argparse.ArgumentParser(description='Paired SUMO evaluation; defaults to three small scenarios')
    p.add_argument('--config',default='configs/default.yaml');p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--seed',type=int,default=200000);p.add_argument('--checkpoint')
    p.add_argument('--output',default='results/smoke');p.add_argument('--stress',action='store_true')
    args=p.parse_args()
    result=evaluate(load_config(args.config),args.episodes,args.seed,args.output,args.checkpoint,args.stress)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
