import _bootstrap
import argparse
from railway_rl.evaluation.plots import generate_plots


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/smoke');p.add_argument('--training')
    a=p.parse_args()
    for path in generate_plots(a.results,a.training):print(path)

if __name__=='__main__':main()
