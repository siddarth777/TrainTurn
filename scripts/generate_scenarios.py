import _bootstrap
import argparse
from pathlib import Path
from railway_rl.config import load_config
from railway_rl.delays import generate_scenario
from railway_rl.timetable import save_scenario, load_timetable
from dataclasses import replace
from railway_rl.station import Disturbance


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--seed',type=int,default=200000);p.add_argument('--count',type=int,default=3)
    p.add_argument('--output',default='data/processed');p.add_argument('--csv',help='Optional documented timetable; zero exogenous delays')
    a=p.parse_args()
    if a.count<1: p.error('--count must be positive')
    for i in range(a.count):
        case=generate_scenario(load_config(a.config),a.seed+i)
        if a.csv:
            trains=load_timetable(a.csv)
            case=replace(case,trains=trains,disturbances=tuple(Disturbance() for _ in trains),
                         provenance=f'User-supplied CSV: {a.csv}; origin must be documented separately')
        destination=Path(a.output)/f'scenario_{case.seed}.json'
        if destination.exists():raise FileExistsError(destination)
        save_scenario(case,destination)
        print(destination)

if __name__=='__main__':main()
