import torch.nn as nn

# print("DP")
def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.DataParallel))


class MultiGPU(nn.DataParallel):
    """Wraps a network to allow simple multi-GPU training."""
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
            
        except:
            pass
        return getattr(self.module, item)


##
# print("FSDP")
# pip install fairscale
# from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

# def is_multi_gpu(net):
#     return isinstance(net, FSDP)

# class MultiGPU(nn.DataParallel):
#     def __init__(self, module, dim=0, **kwargs):
#         super(MultiGPU, self).__init__(module, dim=dim, **kwargs)

        
#     def __getattr__(self, item):
#         try:
#             return super().__getattr__(item)
#         except AttributeError:
#             pass
#         return getattr(self.module, item)