import os
import numpy as np
import time

np.set_printoptions(suppress=True)


def kld(a1, a2) :
    #(B, *, A), #(B, *, A)
    a1 = np.clip(a1, 0, 1)
    a2 = np.clip(a2, 0, 1)
    log_a1 = np.log(a1 + 1e-10)
    log_a2 = np.log(a2 + 1e-10)
    kld_v = a1 * (log_a1 - log_a2)

    return kld_v.sum(-1)

def jsd(p, q) :
    m = 0.5 * (p + q)
    jsd_v = 0.5 * (kld(p, m) + kld(q, m))

    return jsd_v

def get_latest_model(dirname) :
    dirs = [d for d in os.listdir(dirname) if 'evaluate.json' in os.listdir(os.path.join(dirname, d))]
    if len(dirs) == 0 :
        return None
    max_dir = max(dirs, key=lambda s : time.strptime(s.replace('_', ' ')))
    return os.path.join(dirname, max_dir)
