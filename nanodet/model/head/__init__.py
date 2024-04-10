import copy

from nanodet.model.head.nanodet_head_refine import NanoDetHeadRF
from nanodet.model.head.nanodet_head_refine_cu import NanoDetHeadRFCU
from nanodet.model.head.nanodet_head_refine_cu_dn import NanoDetHeadRFCUDN
from nanodet.model.head.simple_conv_head_at import SimpleConvHeadAT

from .gfl_head import GFLHead
from .nanodet_head import NanoDetHead
from .nanodet_plus_head import NanoDetPlusHead
from .simple_conv_head import SimpleConvHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "NanoDetHeadRF":
        return NanoDetHeadRF(**head_cfg)
    elif name == "NanoDetHeadRFCU":
        return NanoDetHeadRFCU(**head_cfg)
    elif name == "NanoDetHeadRFCUDN":
        return NanoDetHeadRFCUDN(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    elif name == "SimpleConvHeadAT":
        return SimpleConvHeadAT(**head_cfg)
    else:
        raise NotImplementedError
