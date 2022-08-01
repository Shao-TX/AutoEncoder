#%%
import torch

def idx2onehot(label_batch, label_num):
    assert torch.max(label_batch).item() < label_num, "Maximum of labels is out of range"

    label_batch = label_batch.unsqueeze(1)

    onehot_base = torch.zeros(label_batch.size(0), label_num).to(label_batch.device)
    onehot = onehot_base.scatter_(1, label_batch, 1)

    return onehot

#%%