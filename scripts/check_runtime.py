import _bootstrap
import json
import shutil
import torch
import libsumo
from railway_rl.agents.double_dqn import require_cuda
from railway_rl.sumo_network import binary


def main():
    print(json.dumps({'torch':str(torch.__version__),'sumo':libsumo.getVersion()[1],
                      'netconvert':binary('netconvert'),'cuda_available':torch.cuda.is_available()},indent=2))
    device=require_cuda()
    x=torch.ones(4,device=device)
    assert (x*2).tolist()==[2,2,2,2]
    print('CUDA arithmetic passed on',torch.cuda.get_device_name())

if __name__=='__main__':main()
