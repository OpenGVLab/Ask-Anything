import torch
import torch.distributed as dist
from utils.distributed import get_rank, is_dist_avail_and_initialized, is_main_process
import random
import logging

logger = logging.getLogger(__name__)


class MetaLoader(object):
    """ wraps multiple data loader """
    def __init__(self, name2loader):
        """Iterates over multiple dataloaders, it ensures all processes
        work on data from the same dataloader. This loader will end when
        the shorter dataloader raises StopIteration exception.

        loaders: Dict, {name: dataloader}
        """
        self.name2loader = name2loader
        self.name2iter = {name: iter(l) for name, l in name2loader.items()}
        name2index = {name: idx for idx, (name, l) in enumerate(name2loader.items())}
        index2name = {v: k for k, v in name2index.items()}

        iter_order = []
        for n, l in name2loader.items():
            iter_order.extend([name2index[n]]*len(l))

        random.shuffle(iter_order)
        iter_order = torch.Tensor(iter_order).to(torch.device("cuda")).to(torch.uint8)

        # sync
        if is_dist_avail_and_initialized():
            # make sure all processes have the same order so that
            # each step they will have data from the same loader
            dist.broadcast(iter_order, src=0)
        self.iter_order = [index2name[int(e.item())] for e in iter_order.cpu()]

        logger.info(str(self))

    def __str__(self):
        output = [f"MetaLoader has {len(self.name2loader)} dataloaders, {len(self)} batches in total"]
        for idx, (name, loader) in enumerate(self.name2loader.items()):
            output.append(
                f"dataloader index={idx} name={name}, batch-size={loader.batch_size} length(#batches)={len(loader)} "
            )
        return "\n".join(output)

    def __len__(self):
        return len(self.iter_order)

    def __iter__(self):
        """ this iterator will run indefinitely """
        for name in self.iter_order:
            _iter = self.name2iter[name]
            batch = next(_iter)
            yield name, batch


class MetaLoader_rs(object):
    """ wraps multiple data loader """
    def __init__(self, name2loader, skip_num=0):
        """Iterates over multiple dataloaders, it ensures all processes
        work on data from the same dataloader. This loader will end when
        the shorter dataloader raises StopIteration exception.

        loaders: Dict, {name: dataloader}
        """
        self.name2loader = name2loader
        name2index = {name: idx for idx, (name, l) in enumerate(name2loader.items())}
        index2name = {v: k for k, v in name2index.items()}

        iter_order = []
        for n, l in name2loader.items():
            iter_order.extend([name2index[n]]*len(l))

        random.shuffle(iter_order)
        iter_order = torch.Tensor(iter_order).to(torch.device("cuda")).to(torch.uint8)

        # sync
        if is_dist_avail_and_initialized():
            # make sure all processes have the same order so that
            # each step they will have data from the same loader
            dist.broadcast(iter_order, src=0)
        
        if skip_num > 0:
            iter_order_skip = iter_order[:skip_num]
            for k, v in index2name.items():
                media_step = (iter_order_skip == k).sum().item()
                name2loader[v].sampler.set_start_iter(media_step)
                logger.info(f"{v} dataloder skip steps: {media_step}")
            iter_order = iter_order[skip_num:]
            self.name2loader = name2loader
        else:
            logger.info("Do not skip steps for any dataloader!")
            for k, v in index2name.items():
                name2loader[v].sampler.set_start_iter(0)
                
        self.name2iter = {name: iter(l) for name, l in name2loader.items()}
        self.iter_idx = iter_order
        self.iter_order = [index2name[int(e.item())] for e in iter_order.cpu()]
        
        logger.info(str(self))

    def __str__(self):
        output = [f"MetaLoader has {len(self.name2loader)} dataloaders, {len(self)} batches in total"]
        for idx, (name, loader) in enumerate(self.name2loader.items()):
            length = (self.iter_idx == idx).sum()
            output.append(
                f"dataloader index={idx} name={name}, batch-size={loader.batch_size} length(#batches)={length} "
            )
        return "\n".join(output)

    def __len__(self):
        return len(self.iter_order)

    def __iter__(self):
        """ this iterator will run indefinitely """
        for name in self.iter_order:
            _iter = self.name2iter[name]
            batch = next(_iter)
            yield name, batch