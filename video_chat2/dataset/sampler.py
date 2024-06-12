import numpy as np
import torch
import logging
from torch.utils.data.distributed import DistributedSampler


# stolen from https://github.com/facebookresearch/vissl/blob/94def58538d3c7037f5e093196494331eea1a2a2/vissl/data/data_helper.py#L93
class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size=None, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, shuffle=False, seed=seed)

        self.start_iter = 0
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas
        print(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        indices = np.array(
            list(
                range(
                    (self.rank * self.num_samples), (self.rank + 1) * self.num_samples
                )
            )
        )[shuffling].tolist()

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter
