import torch
import glob
import os

name = 'MOSEI_0.3_0.1'

ckpts = sorted(glob.glob(os.path.join("ckpt/"+name, 'best*')), reverse=True)
if not os.path.exists("ckpt/"+name+"_RENAME"):
    os.makedirs("ckpt/"+name+"_RENAME")

for ckpt in ckpts:
    args = torch.load(ckpt)['args']
    if args.model == 'Model_MCAN':
        args.model = 'Model_MAT'
    else:
        args.model = 'Model_MNT'

    state = {
        'state_dict': torch.load(ckpt)['state_dict'],
        'optimizer': torch.load(ckpt)['optimizer'],
        'args': args,
    }
    torch.save(
        state,
        os.path.join("ckpt/"+name+"_RENAME", os.path.basename(ckpt))

    )



