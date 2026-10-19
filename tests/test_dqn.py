import numpy as np
import pytest
import torch
from railway_rl.agents.double_dqn import DoubleDQN, bellman_targets, require_cuda
from railway_rl.config import load_config


def test_cuda_requirement_is_explicit():
    if torch.cuda.is_available():
        assert require_cuda().type=='cuda'
    else:
        with pytest.raises(RuntimeError,match='no CPU fallback'):
            require_cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-only neural tests: no accessible GPU')
def test_masked_double_target_and_terminal():
    def tensor(x,dtype=None):return torch.tensor(x,device='cuda',dtype=dtype)
    rewards=tensor([1.,2.,3.]); terminals=tensor([False,True,False])
    masks=tensor([[True,False,True],[False,False,False],[True,True,False]])
    online=tensor([[4.,100.,5.],[0.,0.,0.],[8.,7.,90.]])
    target=tensor([[90.,900.,10.],[float('nan')]*3,[6.,80.,900.]])
    y=bellman_targets(rewards,terminals,tensor([2.,1.,3.]),masks,online,target,0.5)
    torch.testing.assert_close(y,tensor([3.5,2.,3.75]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-only neural tests: no accessible GPU')
def test_optimizer_mask_and_checkpoint(tmp_path):
    c=load_config(); c['agent'].update(batch_size=2,warmup=2,train_every=1,target_update=1,replay_capacity=8)
    agent=DoubleDQN(4,3,c,2)
    usage = agent.memory_usage()
    total = torch.cuda.get_device_properties(agent.device).total_memory
    fraction = torch.cuda.memory.get_per_process_memory_fraction(agent.device)
    assert fraction * total / 2**30 == pytest.approx(usage['allocator_limit_gib'])
    assert usage['allocated_gib'] < usage['allocator_limit_gib']
    mask=np.array([False,True,False]);obs=np.zeros(4,np.float32)
    from copy import deepcopy
    rng_before = deepcopy(agent.rng.bit_generator.state)
    agent.act(obs, mask)
    assert agent.rng.bit_generator.state == rng_before
    assert all(agent.act(obs,mask,epsilon)==1 for epsilon in [0,1])
    before=[p.clone() for p in agent.online.parameters()]
    for i in range(2):agent.replay.add(obs,1,-1,obs,False,False,1,mask)
    assert agent.learn()['loss']>=0
    assert any(not torch.equal(a,b) for a,b in zip(before,agent.online.parameters()))
    agent.save(tmp_path/'agent.pt',{'test':True})
    restored=DoubleDQN.load(tmp_path/'agent.pt')
    assert restored.act(obs,mask)==1
    for a,b in zip(agent.online.parameters(),restored.online.parameters()):torch.testing.assert_close(a,b)
