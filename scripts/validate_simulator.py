import _bootstrap
from copy import deepcopy
from railway_rl.config import load_config
from railway_rl.delays import generate_scenario
from railway_rl.baselines import POLICIES
from railway_rl.evaluation.evaluator import run_episode


def main():
    c=load_config();c['scenario']['severity']=0
    case=generate_scenario(c,7)
    metrics,_,_=run_episode(c,case,lambda o,m:POLICIES['timetable'](o,m))
    print('Zero-disruption timetable:',metrics)
    if metrics['processed_fraction']!=1 or metrics['weighted_station_delay']>1e-6:
        raise AssertionError('Nominal SUMO timetable is not delay-free; repair feasibility before training')
    assert metrics['safety_violations']==0
    print('Nominal feasibility and safety checks passed')

if __name__=='__main__':main()
