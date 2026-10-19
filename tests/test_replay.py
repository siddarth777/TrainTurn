import numpy as np
import pytest
from railway_rl.agents.replay_buffer import ReplayBuffer


def test_replay_wraparound_masks_and_seed():
    buffers=[ReplayBuffer(3,2,2,5) for _ in range(2)]
    for b in buffers:
        for i in range(5):
            b.add(np.array([i,i]),0,float(i),np.array([i+1,i+1]),False,True,2,np.array([True,False]))
    assert buffers[0].size==3
    samples=[b.sample(3) for b in buffers]
    assert set(samples[0]['rewards'])=={2,3,4}
    for key in samples[0]:np.testing.assert_array_equal(samples[0][key],samples[1][key])
    assert samples[0]['truncated'].all() and not samples[0]['terminated'].any()
    with pytest.raises(ValueError):buffers[0].sample(4)
