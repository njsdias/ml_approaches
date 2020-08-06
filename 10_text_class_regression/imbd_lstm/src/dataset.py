import torch


class IMBDDataset:
    """
    This class returns one sample of the training or validation data.
    :param reviews: this is a numpy array
    :param targets: a vector, numpy array
    """
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        # returns length of the dataset
        return len(self.reviews)

    def __getitem__(self, item):
        # for any given item, which is an int,
        # return review and targets as torch tensor
        # item is the index of the item in concern

        review = self.reviews[item, :]
        target = self.target[item]

        return {
            "review": torch.tensor(review, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }

