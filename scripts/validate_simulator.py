import _bootstrap
import argparse
from railway_rl.config import load_config
from railway_rl.delays import generate_scenario
from railway_rl.baselines import POLICIES
from railway_rl.evaluation.evaluator import run_episode


def main():
    p=argparse.ArgumentParser(description='Zero-disruption timetable must be safe and delay-free in SUMO')
    p.add_argument('--config',default='configs/default.yaml');p.add_argument('--seed',type=int,default=7)
    a=p.parse_args()
    c=load_config(a.config);c['scenario']['severity']=0
    case=generate_scenario(c,a.seed)
    metrics,_,_=run_episode(c,case,lambda o,m:POLICIES['timetable'](o,m,c['environment']['max_platforms']))
    print(f'Zero-disruption timetable ({a.config}, seed {a.seed}):',metrics)
    if metrics['processed_fraction']!=1 or metrics['weighted_station_delay']>1e-6:
        raise AssertionError('Nominal SUMO timetable is not delay-free; repair feasibility before training')
    assert metrics['safety_violations']==0
    print('Nominal feasibility and safety checks passed')

if __name__=='__main__':main()
