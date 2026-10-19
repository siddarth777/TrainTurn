from dataclasses import replace
import pytest
from railway_rl.environment import TrainTurnEnv
from railway_rl.config import load_config
from railway_rl.station import Closure, Disturbance, Train
from railway_rl.sumo_simulator import SumoSimulator
from railway_rl.baselines import POLICIES
from railway_rl.evaluation.metrics import episode_metrics
from test_simulator import scenario


def test_sumo_physics_and_clearance():
    case = scenario((Train('T1','express',0,10,1), Train('T2','express',0,15,1)))
    sim = SumoSimulator(case)
    try:
        sim.assign(1)
        assert not sim.feasible(sim.current, 1)
        for _ in range(50):
            if sim.current and sim.feasible(sim.current,1):
                sim.assign(1)
            sim.tick()
            if sim.complete:break
        assert sim.complete
        assert sim.states['T1'].station_departure >= 10
        assert sim.states['T1'].departure > sim.states['T1'].entry
        sim.assert_safety()
    finally:sim.close()


def test_sumo_small_episode_and_determinism():
    c=load_config(); c['scenario']['trains']=5; c['scenario']['severity']=0
    histories=[]
    for _ in range(2):
        env=TrainTurnEnv(c)
        try:
            obs,info=env.reset(seed=19)
            for step in range(c['environment']['horizon']):
                action=POLICIES['timetable'](obs,info['action_mask'])
                obs,reward,term,trunc,info=env.step(action)
                assert env.observation_space.contains(obs)
                if term or trunc:break
            assert term and not trunc
            metrics,rows=episode_metrics(env.sim,c['environment']['horizon'])
            assert metrics['safety_violations']==0
            histories.append((metrics,rows))
        finally:env.close()
    assert histories[0]==histories[1]


def test_sumo_hold_limit_and_truncation():
    c=load_config();c['environment']['horizon']=3;c['environment']['hold_limit']=1
    case=scenario(closures=tuple(Closure(i,0,10) for i in range(1,4)))
    env=TrainTurnEnv(c,case)
    try:
        env.reset()
        for _ in range(3):
            assert env.action_masks()[0] and not env.action_masks()[1:].any()
            _,_,term,trunc,info=env.step(0)
        assert trunc and not term and len(info['unfinished_trains'])==1
        env.sim.assert_safety()
    finally:env.close()


@pytest.mark.parametrize('platforms,trains', [(3, 15), (8, 16)])
def test_nominal_sumo_plan_is_feasible(platforms, trains):
    from railway_rl.delays import generate_scenario
    from railway_rl.evaluation.evaluator import run_episode
    c=load_config();c['scenario'].update(severity=0, trains=trains)
    c['station']['platform_count'] = platforms
    case = generate_scenario(c, 7)
    assert all(any(p.compatible(t) for t in case.trains) for p in case.station.platforms)
    metrics,rows,trace=run_episode(c,case,lambda o,m:POLICIES['timetable'](o,m))
    assert metrics['weighted_station_delay']==0
    assert metrics['mean_departure_lateness_completed']==0
    assert metrics['processed_fraction']==1
    assert len(trace)==trains and all(row['end'] is not None for row in trace)


def test_discretionary_holds_exclude_forced_waits():
    from railway_rl.evaluation.evaluator import run_episode
    c = load_config()
    first = True

    def policy(obs, mask):
        nonlocal first
        if first:
            first = False
            return 0
        return POLICIES['fcfs'](obs, mask)

    metrics, _, _ = run_episode(c, scenario(), policy)
    assert metrics['discretionary_holds'] == 1
    closed = scenario(closures=tuple(Closure(i, 0, 3) for i in range(1, 4)))
    metrics, _, _ = run_episode(c, closed, lambda o, m: POLICIES['fcfs'](o, m))
    assert metrics['holds'] > 0
    assert metrics['discretionary_holds'] == 0
