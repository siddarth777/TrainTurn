import pytest
from railway_rl.evaluation.statistics import summarize, paired_difference


def test_paired_bootstrap_uses_scenario_pairs():
    a={3:13,1:11,2:12};b={1:10,2:11,3:12}
    result=paired_difference(a,b)
    assert result['mean']==result['ci95_low']==result['ci95_high']==1
    with pytest.raises(ValueError):paired_difference(a,{1:1})
    assert summarize([1,2,8],seed=2)==summarize([1,2,8],seed=2)
    with pytest.raises(ValueError):summarize([float('nan')])
