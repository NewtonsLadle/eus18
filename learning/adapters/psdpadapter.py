import numpy as np
import random
from pathlib import Path
import sys

from adapters.medsegadapter import MedSegAdapter

class PSDPAdapter(MedSegAdapter):
    """Adapter for Pancreas Segmentation Data-set (ignoring tumors for now)

    Keeps track of number of bundles stored on disk, and frequencies of the
    classes. Returns a bundle of the requested type on command
    """

    def __init__(self, psdp_dir, intensities=True, locations=True, trn_lim=-1, prefix='axial'):
        """Constructor for the PSD Adapter object

        Takes root psd directory on this machine and optionally a limit
        on how many bundles to use
        """

        if prefix == 'axial':
            trn = 256
            trt = 453
            van = 17
            vat = 29
            height = 512
            width = 512
        elif prefix == 'sagittal':
            trn = 478
            trt = 1033
            van = 33
            vat = 71
            height = 466
            width = 512
        elif prefix == 'coronal':
            trn = 268
            trt = 1243
            van = 16
            vat = 87
            height = 466
            width = 512
        else:
            print("unsupported prefix")
            sys.exit()


        channels = 1
        super().__init__(
            data_dir=psdp_dir,
            height=height,
            width=width,
            frequencies=[0.99528459, 0.00471541],
            bundle_size=20,
            x_n_pfx = prefix+'-X-ntl-',
            y_n_pfx = prefix+'-Y-ntl-',
            l_n_pfx = prefix+'-L-ntl-',
            x_t_pfx = prefix+'-X-tvl-',
            y_t_pfx = prefix+'-Y-tvl-',
            l_t_pfx = prefix+'-L-tvl-',
            trn_ntl_batches=trn,
            trn_tvl_batches=trt,
            val_ntl_batches=van,
            val_tvl_batches=vat,
            trn_lim=trn_lim,
            in_channels=channels,
            intensities=True,
            locations=False
        )
