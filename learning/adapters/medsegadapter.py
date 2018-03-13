import numpy as np
import random
from pathlib import Path

class MedSegAdapter(object):
    """Adapter for Pancreas Segmentation Data-set (ignoring tumors for now)

    Keeps track of number of bundles stored on disk, and frequencies of the
    classes. Returns a bundle of the requested type on command
    """

    def __init__(self, data_dir, height, width, frequencies, bundle_size,
                 x_n_pfx, y_n_pfx, l_n_pfx, x_t_pfx, y_t_pfx, l_t_pfx,
                 trn_ntl_batches, trn_tvl_batches, val_ntl_batches,
                 val_tvl_batches, trn_lim=-1, in_channels=1, intensities=True,
                 locations=True):
        """Constructor for the PSD Adapter object

        Takes root psd directory on this machine and optionally a limit
        on how many bundles to use
        """
        # Dataset constants
        self.in_channels = in_channels
        self.out_channels = 2

        self.height = height
        self.width = width

        self.frequencies = frequencies

        self.bundle_size = bundle_size

        self.base_dir = Path(data_dir)
        self.train_dir = self.base_dir / 'training'
        self.val_dir = self.base_dir / 'validation'
        self.x_n_pfx = x_n_pfx
        self.y_n_pfx = y_n_pfx
        self.l_n_pfx = l_n_pfx
        self.x_t_pfx = x_t_pfx
        self.y_t_pfx = y_t_pfx
        self.l_t_pfx = l_t_pfx

        if trn_lim < 0:
            self.trn_ntl_batches = trn_ntl_batches
            self.trn_tvl_batches = trn_tvl_batches
        else:
            self.trn_ntl_batches = trn_lim
            self.trn_tvl_batches = trn_lim

        self.val_ntl_batches = val_ntl_batches
        self.val_tvl_batches = val_tvl_batches

        self.intensities = intensities
        self.locations = locations


    def _get_bundle(self, loc, xpfx, ypfx, lpfx, i):
        """Load a bundle from disk

        Provide location, prefixes, and index
        """
        X = np.zeros(
            [self.bundle_size,self.height, self.width, 0],
            np.float32
        )
        if self.intensities:
            X = np.concatenate(
                (
                    X,
                    np.load(
                        str(loc / (xpfx + str(i) + '.npy'))
                    )
                ),
                axis=3
            )
        if self.locations:
            X = np.concatenate(
                (
                    X,
                    np.load(
                        str(loc / (lpfx + str(i) + '.npy'))
                    )
                ),
                axis=3
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
            lpfx = self.l_n_pfx
        else:
            pick = int(r*self.trn_tvl_batches)
            xpfx = self.x_t_pfx
            ypfx = self.y_t_pfx
            lpfx = self.l_t_pfx

        return self._get_bundle(self.train_dir, xpfx, ypfx, lpfx, pick)



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
        lpfx = self.l_n_pfx
        for i in range(0, self.val_ntl_batches):
            (
                nX[self.bundle_size*i:self.bundle_size*(i+1),:,:,:],
                nY[self.bundle_size*i:self.bundle_size*(i+1),:,:,:]
            ) = self._get_bundle(self.val_dir, xpfx, ypfx, lpfx, i)

        xpfx = self.x_t_pfx
        ypfx = self.y_t_pfx
        lpfx = self.l_t_pfx
        for i in range(0, self.val_tvl_batches):
            (
                tX[self.bundle_size*i:self.bundle_size*(i+1),:,:,:],
                tY[self.bundle_size*i:self.bundle_size*(i+1),:,:,:]
            ) = self._get_bundle(self.val_dir, xpfx, ypfx, lpfx, i)

        return ((nX, nY), (tX, tY))

    #TODO TESTING!
