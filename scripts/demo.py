import _bootstrap
import argparse
import json
from railway_rl.baselines import POLICIES
from railway_rl.config import load_config
from railway_rl.delays import generate_scenario
from railway_rl.evaluation.evaluator import run_episode
from railway_rl.timetable import load_scenario


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--policy',choices=[*POLICIES,'dqn'],default='timetable');p.add_argument('--seed',type=int,default=123)
    p.add_argument('--checkpoint');p.add_argument('--scenario',help='Replay a saved immutable JSON scenario')
    a=p.parse_args();c=load_config(a.config)
    if a.policy=='dqn':
        if not a.checkpoint:p.error('dqn requires --checkpoint')
        from railway_rl.agents.double_dqn import DoubleDQN
        agent=DoubleDQN.load(a.checkpoint);policy=lambda o,m:agent.act(o,m)
        if agent.config['environment']!=c['environment']:p.error('Checkpoint environment settings differ')
    else:
        policy=lambda o,m:POLICIES[a.policy](o,m,c['environment']['max_platforms'])
    case=load_scenario(a.scenario) if a.scenario else generate_scenario(c,a.seed)
    metrics,trains,occupancy=run_episode(c,case,policy)
    print(json.dumps({'policy':a.policy,'metrics':metrics,'trains':trains,'occupancy':occupancy},indent=2))

if __name__=='__main__':main()
