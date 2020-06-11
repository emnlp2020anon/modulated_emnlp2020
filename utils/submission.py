

def cmumosei_7(a):
    if a == 0:
        res = -3
    if a == 1:
        res = -2
    if a == 2:
        res = -1
    if a == 3:
        res = 0
    if a == 4:
        res = 1
    if a == 5:
        res = 2
    if a == 6:
        res = 3
    return res

import numpy as np
import pickle

y = pickle.load(open("ckpt/Model_MCAN_s_new_cross_full_pipeline_6_512_8dd/private_avg_preds_3.p", "rb"))
for k in y.keys():
    y[k] = [cmumosei_7(y[k])]

x = pickle.load(open("ckpt/Model_MCAN_e_new_cross_full_pipeline_6_512_8/private_avg_preds_4.p", "rb"))
for k in x.keys():
    x[k] = [int(v) for v in x[k]]

for k in y.keys():
    if k not in x:
        print("noooon")
    y[k] = np.array(y[k] + x[k])

pickle.dump(y, open("preds_normal.p","wb"))