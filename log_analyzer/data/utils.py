
"""
Data loading functions
"""

from torch.utils.data import DataLoader

def create_data_loaders():

    train_loader = DataLoader()
    test_loader = DataLoader()

    return train_loader, test_loader