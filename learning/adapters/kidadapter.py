import numpy as np
import random
from pathlib import Path
import sys

from adapters.cvsegadapter import CVSegAdapter

class KidAdapter(CVSegAdapter):
    """Adapter for the Kidney dataset

    Has a "sets" object hard-coded which provides information about loading
    the data
    """

    def __init__(self, sk_dir, intensities=True, locations=True, trn_lim=-1, out="cjweight", plane="axial"):
        """Constructor for the PSD Adapter object

        Takes root psd directory on this machine and optionally a limit
        on how many bundles to use
        """

        if (plane == "sagittal"):
            sets = {
                "cjweight": {
                    "ntl": 10,
                    "tvl": 17
                },
                "nsathian": {
                    "ntl": 10,
                    "tvl": 17
                },
                "ravis071": {
                    "ntl": 8,
                    "tvl": 18
                },
                "trade011": {
                    "ntl": 8,
                    "tvl": 19
                },
                "srkrishn": {
                    "ntl": 8,
                    "tvl": 19
                },
                "blake367": {
                    "ntl": 9,
                    "tvl": 17
                },
            }

            height = 512
            width = 100

        elif (plane == "coronal"):
            sets = {
                "cjweight": {
                    "ntl": 6,
                    "tvl": 20
                },
                "nsathian": {
                    "ntl": 6,
                    "tvl": 20
                },
                "ravis071": {
                    "ntl": 4,
                    "tvl": 22
                },
                "trade011": {
                    "ntl": 5,
                    "tvl": 21
                },
                "srkrishn": {
                    "ntl": 5,
                    "tvl": 21
                },
                "blake367": {
                    "ntl": 6,
                    "tvl": 21
                },
            }

            height = 100
            width = 512

        else:
            plane = "axial"
            sets = {
                "cjweight": {
                    "ntl": 3,
                    "tvl": 3
                },
                "nsathian": {
                    "ntl": 3,
                    "tvl": 3
                },
                "ravis071": {
                    "ntl": 4,
                    "tvl": 2
                },
                "trade011": {
                    "ntl": 4,
                    "tvl": 2
                },
                "srkrishn": {
                    "ntl": 3,
                    "tvl": 3
                },
                "blake367": {
                    "ntl": 4,
                    "tvl": 2
                },
            }

            height = 512
            width = 512


        if out != "cjweight":
            val = "cjweight"
        else:
            val = "trade011"

        super().__init__(
            data_dir=sk_dir,
            height=height,
            width=width,
            bundle_size=20,
            in_channels=1,
            sets=sets,
            out=out,
            val=val,
            plane=plane
        )
