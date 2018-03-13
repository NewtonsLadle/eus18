import numpy as np
from pathlib import Path
import random

class LiSAdapter(object):
    """Adapter for Liver Segmentation Data-set (ignoring tumors for now)

    Keeps track of number of bundles stored on disk, and frequencies of the
    classes. Returns a bundle of the requested type on command
    """

    def __init__(self, lis_dir, trn_lim=-1):
        """Constructor for the LiS Adapter object

        Takes root lis directory on this machine and optionally a limit
        on how many bundles to use
        """

        # Dataset constants
        self.in_channels = 1
        self.out_channels = 2
        self.height = 512
        self.width = 512

        # TODO standardize this to be 1 for all sets
        self.rare_class = 0

        # TODO read this from disk (?)
        self.frequencies = [0.07, 0.93]

        # Number of slices returned per load call
        self.bundle_size = 20

        # Operating system calls
        self.base_dir = Path(lis_dir)
        self.train_dir = self.base_dir / 'training'
        self.val_dir = self.base_dir / 'validation'

        self.x_n_pfx = 'x-ntl-'
        self.y_n_pfx = 'y-ntl-lo-'
        self.x_t_pfx = 'x-tvl-'
        self.y_t_pfx = 'y-tvl-lo-'

        if (trn_lim == -1):
            self.trn_ntl_batches = 533
            self.trn_tvl_batches = 1194
        else:
            self.trn_ntl_batches = trn_lim
            self.trn_tvl_batches = trn_lim

        #TODO reorganize set so this is actually smaller
        #self.val_ntl_batches = 227
        #self.val_tvl_batches = 433
        self.val_ntl_batches = 10
        self.val_tvl_batches = 20


    def _get_bundle(self, loc, xpfx, ypfx, i):
        """Load a bundle from disk

        Provide location, prefixes, and index
        """
        X = np.load(
            str(loc / (xpfx + str(i) + '.npy'))
        )
        Y = np.load(
            str(loc / (ypfx + str(i) + '.npy'))
        )
        return X, Y


    def get_training_bundle(self, r, nontrivial=True):
        """Load a training bundle from disk

        Returns bundle from training set with index proportional to r
        """
        if nontrivial:
            pick = int(r*self.trn_ntl_batches)
            xpfx = self.x_n_pfx
            ypfx = self.y_n_pfx
        else:
            pick = int(r*self.trn_tvl_batches)
            xpfx = self.x_t_pfx
            ypfx = self.y_t_pfx

        return self._get_bundle(self.train_dir, xpfx, ypfx, pick)



    def get_validation_pool(self):
        """Load the entire validation pool from disk

        Returns tuple of tuples((nX, nY), (tX, tY))
        Where n corresponds to nontrivial data, and t corresponds to trivial data
        """

        ntl_num_slices = self.val_ntl_batches*self.bundle_size
        tvl_num_slices = self.val_tvl_batches*self.bundle_size

        nX = np.zeros(
            (ntl_num_slices, self.height, self.width, self.in_channels),
            np.float32
        )
        nY = np.zeros(
            (ntl_num_slices, self.height, self.width, 1),
            np.float32
        )
        tX = np.zeros(
            (tvl_num_slices, self.height, self.width, self.in_channels),
            np.float32
        )
        tY = np.zeros(
            (tvl_num_slices, self.height, self.width, 1),
            np.float32
        )

        xpfx = self.x_n_pfx
        ypfx = self.y_n_pfx
        for i in range(0, self.val_ntl_batches):
            (
                nX[self.bundle_size*i:self.bundle_size*(i+1),:,:,:],
                nY[self.bundle_size*i:self.bundle_size*(i+1),:,:,:]
            ) = self._get_bundle(self.val_dir, xpfx, ypfx, i)

        xpfx = self.x_t_pfx
        ypfx = self.y_t_pfx
        for i in range(0, self.val_tvl_batches):
            (
                tX[self.bundle_size*i:self.bundle_size*(i+1),:,:,:],
                tY[self.bundle_size*i:self.bundle_size*(i+1),:,:,:]
            ) = self._get_bundle(self.val_dir, xpfx, ypfx, i)

        return ((nX, nY), (tX, tY))

    #TODO TESTING!
