import _bootstrap
import argparse
import json
from railway_rl.config import load_config, validate
from railway_rl.training import train


def main():
    p=argparse.ArgumentParser(description='Small CUDA-only Double DQN training on SUMO')
    p.add_argument('--config',default='configs/default.yaml');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--episodes',type=int);p.add_argument('--output',default='checkpoints/smoke')
    p.add_argument('--warm-start',help='Load weights; restart optimizer, exploration and replay')
    args=p.parse_args();c=load_config(args.config)
    if args.episodes is not None:c['training']['episodes']=args.episodes
    print(json.dumps(train(validate(c),args.seed,args.output,args.warm_start),indent=2))

if __name__=='__main__':main()
