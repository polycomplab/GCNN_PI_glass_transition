import copy

import torch


def split_train_val(dataset, test_size: int):
    """Shuffles the dataset and splits it
    into a train set and a val set"""
    
    print('len(dataset)', len(dataset))
    train_dataset = copy.deepcopy(dataset)
    indices = train_dataset.mol_indices
    indices = indices[torch.randperm(len(indices))]

    if test_size > len(indices):
        raise ValueError(f'dataset is too small ({len(indices)} mols) '
                         f'for the desired test size ({test_size} mols)')
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    train_indices = train_indices.sort().values
    test_indices = test_indices.sort().values
    
    test_dataset = copy.deepcopy(dataset)
    train_dataset.mol_indices = train_indices
    test_dataset.mol_indices = test_indices
    return train_dataset, test_dataset


def split_subindex(dataset, subindex_size: int):
    """Shuffles mols and reduces dataset size"""

    mol_indices = copy.copy(dataset.mol_indices)
    mol_indices = mol_indices[torch.randperm(len(mol_indices))]
    mol_indices = mol_indices[-subindex_size:]
    mol_indices = mol_indices.sort().values
    subset = copy.deepcopy(dataset)
    subset.mol_indices = mol_indices
    return subset


def split_train_subset(dataset, train_size: int, max_train_size: int):
    '''For training on varying subsets ("train_size") of the dataset
    with keeping the test size same'''

    print('len(dataset)', len(dataset))
    test_size = len(dataset) - max_train_size
    assert test_size > 0
    print('test size', test_size)
    train_dataset = copy.deepcopy(dataset)
    indices = train_dataset.mol_indices
    indices = indices[torch.randperm(len(indices))]

    # train_indices = indices[:-test_size]
    train_indices = indices[:train_size]
    test_indices = indices[-test_size:]
    
    train_indices = train_indices.sort().values
    test_indices = test_indices.sort().values
    
    test_dataset = copy.deepcopy(dataset)
    train_dataset.mol_indices = train_indices
    test_dataset.mol_indices = test_indices
    return train_dataset, test_dataset


def k_fold_split_fixed(dataset, k: int):
    indices = dataset.mol_indices
    indices = indices[torch.randperm(len(indices))]
    
    total = 0
    train_size = len(indices)//k * (k-2)
    for i in range(k):
        test_size = (len(indices)-train_size)//2
        val_size = len(indices)-train_size-test_size
        testval_size = test_size + val_size

        test_start = i*test_size
        test_end = i*test_size + test_size
        val_end = (test_end + val_size) % len(indices)

        testval = indices[i*test_size:i*test_size+testval_size]
        if i*test_size+testval_size > len(indices):
            testval_extras = i*test_size+testval_size - len(indices)
            testval = torch.cat([testval, indices[:testval_extras]])
        test_split = testval[:test_size].sort().values
        val_split = testval[test_size:].sort().values

        if val_end < test_start:
            train_before_start = val_end
            train_after_start = len(indices)
        else:
            train_before_start = 0
            train_after_start = val_end

        train_split = torch.cat([
            indices[train_before_start:test_start],
            indices[train_after_start:len(indices)]]).sort().values

        print(f'test split {i}: num: {len(test_split)}, values:\n', test_split)
        print(f'val split {i}: num: {len(val_split)}, values:\n', val_split)
        total += len(test_split)
        # print(total)
        train_dataset = copy.deepcopy(dataset)
        # train_dataset.mol_indices = torch.cat(split_i).sort().values
        train_dataset.mol_indices = train_split
        print(f'train split {i}: num: {len(train_split)}, values:\n',
              train_dataset.mol_indices)
        print()
        test_dataset = copy.deepcopy(dataset)
        test_dataset.mol_indices = test_split
        val_dataset = copy.deepcopy(dataset)
        val_dataset.mol_indices = val_split
        yield (train_dataset, val_dataset, test_dataset)


def k_fold_split(dataset, k: int):
    indices = dataset.mol_indices
    indices = indices[torch.randperm(len(indices))]
    print('num mols:', len(indices))
    split_indices = list(indices.split(len(indices)//k))
    print(f'mol indices:', split_indices)
    # total = 0
    for i in range(k+1):
        split_i = copy.deepcopy(split_indices)
        test_split = split_i.pop(i).sort().values
        val_split = split_i.pop(i%k).sort().values
        # total += len(test_split)
        # print(total)
        train_dataset = copy.deepcopy(dataset)
        train_dataset.mol_indices = torch.cat(split_i).sort().values
        test_dataset = copy.deepcopy(dataset)
        test_dataset.mol_indices = test_split
        val_dataset = copy.deepcopy(dataset)
        val_dataset.mol_indices = val_split
        yield (train_dataset, val_dataset, test_dataset)
