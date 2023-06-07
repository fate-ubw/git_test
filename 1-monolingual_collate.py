import torch
from torch.utils.data import DataLoader
import data_utils
import pdb
# define some dummy data
def collate(samples, pad_idx, eos_idx):

    if len(samples) == 0:
        return {}

    def merge(key, is_list=False): #key :source & target
        if is_list: # If is_list is True, merge will merge a list of tokens instead of a single token.
            res = [] 
            for i in range(len(samples[0][key])):
                res.append(data_utils.collate_tokens(
                    [s[key][i] for s in samples], pad_idx, eos_idx, left_pad=False,
                )) 
            return res
        else: #len of list = 1

            return data_utils.collate_tokens( 

                [s[key] for s in samples], pad_idx, eos_idx, left_pad=False,
            )
    src_tokens = merge('source') 
    if samples[0]['target'] is not None:
        is_target_list = isinstance(samples[0]['target'], list) #
        target = merge('target', is_target_list)
    else:
        target = src_tokens

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'nsentences': len(samples),
        'ntokens': sum(len(s['source']) for s in samples),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': torch.LongTensor([
                s['source'].numel() for s in samples
            ]),
        },
        'target': target, 
    }


if __name__ == '__main__':
    data = [
        {'id': 1, 'source': torch.tensor([1, 2, 3]), 'target': torch.tensor([4, 5, 6])}, # the length of source and target has to be same
        {'id': 2, 'source': torch.tensor([7, 8]), 'target': torch.tensor([9, 10])},
        {'id': 3, 'source': torch.tensor([11, 12, 13, 14]), 'target': torch.tensor([15, 16, 17, 18])},
    ]
    # define the special tokens
    pad_idx = 0 #padding value
    eos_idx = -1 #the postion of padding
    pdb.set_trace()
    # create a DataLoader with our data and collate function
    # dataloader = DataLoader(data, batch_size=2, collate_fn=lambda x: collate(x, pad_idx, eos_idx))

    dataloader = collate(data, pad_idx, eos_idx)
    #the function of collate_fn is to form target input
    # iterate over the DataLoader to get batches of data

    print(dataloader)
    for batch in dataloader:
        print(batch)
'''
{'id': tensor([1, 2]), 'nsentences': 2, 'ntokens': 5, 
    'net_input': {'src_tokens': tensor([[1, 2, 3], [7, 8, 0]]),
     'src_lengths': tensor([3, 2])},
      'target': tensor([[ 4,  5,  6], [ 9, 10,  0]])}
'''
