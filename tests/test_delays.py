from railway_rl.config import load_config
from railway_rl.delays import generate_scenario
from railway_rl.timetable import save_scenario, load_scenario


def test_exogenous_scenarios_reproducible_and_roundtrip(tmp_path):
    c=load_config()
    a,b=generate_scenario(c,11),generate_scenario(c,11)
    assert a==b and a!=generate_scenario(c,12)
    assert all(x.incoming_delay>=0 and x.dwell_extension>=0 for x in a.disturbances)
    save_scenario(a,tmp_path/'scenario.json')
    assert a==load_scenario(tmp_path/'scenario.json')


def test_zero_severity_and_modes():
    c=load_config(); c['scenario']['severity']=0
    assert all(x.incoming_delay==x.dwell_extension==0 for x in generate_scenario(c,3).disturbances)
    c['scenario']['mode']='closure'
    assert len(generate_scenario(c,3).closures)==1
    c['scenario']['severity']=1;c['scenario']['mode']='corridor'
    s=generate_scenario(c,3)
    assert all(d.incoming_delay>=60 for t,d in zip(s.trains,s.disturbances) if t.arrival_corridor=='A')
