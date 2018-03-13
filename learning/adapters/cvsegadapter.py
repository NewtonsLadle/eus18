import numpy as np
import random
from pathlib import Path

class CVSegAdapter(object):
    """Adapter for Pancreas Segmentation Data-set (ignoring tumors for now)

    Keeps track of number of bundles stored on disk, and frequencies of the
    classes. Returns a bundle of the requested type on command
    """

    def __init__(self, data_dir, height, width, bundle_size, in_channels,
                 sets, out, val, plane):
        """Constructor for the Cross Validation Segmentation Adapter object

        Takes information on how to provide data from set#TODO
        """
        # Dataset constants
        self.in_channels = in_channels
        self.out_channels = 2

        self.height = height
        self.width = width

        self.plane = plane

        self.sets = sets
        self.num_sets = len(sets)
        self.out = out # Testing patient
        self.val = val

        self.bundle_size = bundle_size

        self.data_dir = Path(data_dir)

        self.data = self._load_data()

        n = 0
        d = 0
        self.trn_ntl_batches = 0
        self.trn_tvl_batches = 0
        self.val_ntl_batches = 0
        self.val_tvl_batches = 0
        self.tst_ntl_batches = 0
        self.tst_tvl_batches = 0


        for key in self.sets:
            if (key == self.out):
                self.tst_ntl_batches = self.tst_ntl_batches + self.sets[key]["ntl"]
                self.tst_tvl_batches = self.tst_tvl_batches + self.sets[key]["tvl"]
            elif (key == self.val):
                self.val_ntl_batches = self.val_ntl_batches + self.sets[key]["ntl"]
                self.val_tvl_batches = self.val_tvl_batches + self.sets[key]["tvl"]
            else:
                n = n + np.sum(self.sets[key]["ntlY"])
                d = d + (self.sets[key]["ntl"])*self.bundle_size*self.height*self.width
                self.trn_ntl_batches = self.trn_ntl_batches + self.sets[key]["ntl"]
                self.trn_tvl_batches = self.trn_tvl_batches + self.sets[key]["tvl"]
        f = n/d
        self.frequencies = [1-f, f]



    def _load_set(self, prefix, num_ntl, num_tvl):
        """Load a bundle from disk

        Takes prefix, number of nontrivial sets belonging to that patient,
        and number of trivial sets belonging to that patient
        """
        ntlX = np.zeros((self.bundle_size*num_ntl, self.height, self.width, 1))
        ntlY = np.zeros((self.bundle_size*num_ntl, self.height, self.width, 1))
        tvlX = np.zeros((self.bundle_size*num_tvl, self.height, self.width, 1))
        tvlY = np.zeros((self.bundle_size*num_tvl, self.height, self.width, 1))

        for i in range(0, num_ntl):
            ntlX[i*self.bundle_size:(i+1)*self.bundle_size] = np.load(
                str(self.data_dir / ('%s-%s-X-ntl-%d.npy' % (self.plane, prefix, i)))
            )
            ntlY[i*self.bundle_size:(i+1)*self.bundle_size] = np.load(
                str(self.data_dir / ('%s-%s-Y-ntl-%d.npy' % (self.plane, prefix, i)))
            )
        for i in range(0, num_tvl):
            tvlX[i*self.bundle_size:(i+1)*self.bundle_size] = np.load(
                str(self.data_dir / ('%s-%s-X-tvl-%d.npy' % (self.plane, prefix, i)))
            )
            tvlY[i*self.bundle_size:(i+1)*self.bundle_size] = np.load(
                str(self.data_dir / ('%s-%s-Y-tvl-%d.npy' % (self.plane, prefix, i)))
            )

        return ntlX, ntlY, tvlX, tvlY


    def _load_data(self):
        """Load all data into a dict in memory

        Iteratively calls _load_set using information from the sets dict
        """
        for key in self.sets:
            (
                self.sets[key]["ntlX"],
                self.sets[key]["ntlY"],
                self.sets[key]["tvlX"],
                self.sets[key]["tvlY"]
            ) = self._load_set(key, self.sets[key]["ntl"], self.sets[key]["tvl"])


    def get_training_bundle(self, r, nontrivial=True):
        """Load a training bundle from disk

        Returns bundle from training set with index proportional to r
        """
        float_ind = (self.num_sets)*r
        int_ind = int(float_ind)
        r = float_ind - int_ind
        i = -1
        for key in self.sets:
            i = i + 1
            if (key != self.out):
                if i == int_ind:
                    break
            elif (i == int_ind):
                int_ind = int_ind + 1

        if nontrivial:
            pick = int(r*((self.sets[key]["ntl"] - 1)*self.bundle_size + 1))
            return (
                self.sets[key]["ntlX"][pick:pick+self.bundle_size],
                self.sets[key]["ntlY"][pick:pick+self.bundle_size]
            )
        else:
            pick = int(r*((self.sets[key]["tvl"] - 1)*self.bundle_size + 1))
            return (
                self.sets[key]["tvlX"][pick:pick+self.bundle_size],
                self.sets[key]["tvlY"][pick:pick+self.bundle_size]
            )



    def get_testing_pool(self):
        """Load the entire testing pool from disk

        Returns tuple of tuples((nX, nY), (tX, tY))
        Where n corresponds to nontrivial data, and t corresponds to trivial data
        """

        return (
            (self.sets[self.out]["ntlX"], self.sets[self.out]["ntlY"]),
            (self.sets[self.out]["tvlX"], self.sets[self.out]["tvlY"]),
        )


    def get_validation_pool(self):
        """Load the entire validation pool from disk

        Returns tuple of tuples((nX, nY), (tX, tY))
        Where n corresponds to nontrivial data, and t corresponds to trivial data
        """

        return (
            (self.sets[self.val]["ntlX"], self.sets[self.val]["ntlY"]),
            (self.sets[self.val]["tvlX"], self.sets[self.val]["tvlY"]),
        )


    def get_full_out(self):
        """Load entire testing patient from disk and return it"""

        X = np.load(
            str(self.data_dir / ('%s-%s-x.npy' % (self.plane, self.out)))
        )
        Y = np.load(
            str(self.data_dir / ('%s-%s-y.npy' % (self.plane, self.out)))
        )

        return X, Y
