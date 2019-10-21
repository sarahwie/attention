def isTrue(obj, attr) :
    return hasattr(obj, attr) and getattr(obj, attr)

import numpy as np
import torch

def get_sorting_index_with_noise_from_lengths(lengths, noise_frac) :
    if noise_frac > 0 :
        noisy_lengths = [x + np.random.randint(np.floor(-x*noise_frac), np.ceil(x*noise_frac)) for x in lengths]
    else :
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BatchHolder() : 
    def __init__(self, data, target_attn=None) :
        maxlen = max([len(x) for x in data])
        self.maxlen = maxlen
        self.B = len(data)

        lengths = []
        expanded = []
        masks = []
        expanded_attn = []

        if target_attn :
            for d, ta in zip(data, target_attn) :
                assert len(d) == len(ta)
                rem = maxlen - len(d)
                expanded.append(d + [0]*rem)
                lengths.append(len(d))
                masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))
                assert len([1] + [0]*(len(d)-2) + [1]*(rem+1)) == len(ta + [0]*rem) == len(d + [0]*rem)
                #also pad target attention:
                expanded_attn.append(ta + [0]*rem)
        else :
            for _, d in enumerate(data) :
                rem = maxlen - len(d)
                expanded.append(d + [0]*rem)
                lengths.append(len(d))
                masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))

        self.lengths = torch.LongTensor(np.array(lengths)).to(device)
        self.seq = torch.LongTensor(np.array(expanded, dtype='int64')).to(device)
        self.masks = torch.ByteTensor(np.array(masks)).to(device)

        self.hidden = None
        self.predict = None
        self.attn = None

        if target_attn :
                self.target_attn = torch.FloatTensor(expanded_attn).to(device)
        self.inv_masks = ~self.masks

    def generate_frozen_uniform_attn(self):
        attn = np.zeros((self.B, self.maxlen))
        inv_l = 1. / (self.lengths.cpu().data.numpy() - 2)
        attn += inv_l[:, None]
        attn = torch.Tensor(attn).to(device)
        attn.masked_fill_(self.masks, 0) 
        return attn

def kld(a1, a2) :
    #(B, *, A), #(B, *, A)
    a1 = torch.clamp(a1, 0, 1)
    a2 = torch.clamp(a2, 0, 1)
    log_a1 = torch.log(a1 + 1e-10)
    log_a2 = torch.log(a2 + 1e-10)

    kld = a1 * (log_a1 - log_a2)
    kld = kld.sum(-1)

    return kld

def jsd(p, q) :
    m = 0.5 * (p + q)
    jsd = 0.5 * (kld(p, m) + kld(q, m)) #for each instance in the batch
    
    return jsd.unsqueeze(-1) #jsd.squeeze(1).sum()
