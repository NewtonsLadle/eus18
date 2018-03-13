import numpy as np
import threading
import time

class MedSegServer(object):
    """A server for binary medical image segmentation

    Retrieves a certain number of trivial and non-trivial stored batches from
    disk, shuffles them, and returns them sequentially

    Knows about class frequencies via the adapter, and takes information
    about the precision/recall of the priors via methods called by the
    trainer, takes information about proportion of trivial samples likewise
    by methods called by the trainer, and is therefore the best object to
    manage the mf-class weights, although some may be zeroed by the trainer


    NOTE: for the kidney project, this is multi-threaded unnecessarily, and
    memory use is very inefficient
    """

    def __init__(self, adapter, batch_order, type_order,
                 viz_rand=[], trivial_prob=0.0, trn_bundles=20):
        """Construct the Medical Image Segmentation Server

        Builds an object which has an adapter and is ready to take queries for
        the next training batch and or the next validation/testing batch
        """
        self.adapter = adapter
        self.trivial_prob = trivial_prob
        self.default_trivial_prob = 1.0 - self.adapter.trn_ntl_batches/self.adapter.trn_tvl_batches
        # the order in which to load batches into memory (gives some but not
        # perfect determinism to training)
        self.batch_order = batch_order
        self.type_order = type_order

        # training bundles to hold in lake at once
        self.trn_bundles = trn_bundles
        # calculate training slices held in memory at once
        self.trn_slices = trn_bundles*adapter.bundle_size
        # Allocate memory for training pool
        trn_X = np.zeros(
            (
                self.trn_slices,
                self.adapter.height,
                self.adapter.width,
                self.adapter.in_channels
            ),
            np.float32
        )
        trn_Y = np.zeros(
            (
                self.trn_slices,
                self.adapter.height,
                self.adapter.width,
                1
            ),
            np.float32
        )
        # Keep track of global step for semi-deterministic batch loading
        self.trn_j = 0
        # Useful to keep this around for reordering pool after loading new data
        self.pool_reorder = np.array(
            np.concatenate(
                [
                    i + np.arange(0, self.trn_slices, self.trn_bundles)
                    for i in range(0, self.trn_bundles)
                ]
            ),
            np.int32
        )

        # Compute information about validation set
        self.ntl_val_slices = self.adapter.val_ntl_batches*self.adapter.bundle_size
        self.tvl_val_slices = self.adapter.val_tvl_batches*self.adapter.bundle_size
        self.val_slices = self.ntl_val_slices + self.tvl_val_slices

        # Compute information about testing set
        self.ntl_tst_slices = self.adapter.tst_ntl_batches*self.adapter.bundle_size
        self.tvl_tst_slices = self.adapter.tst_tvl_batches*self.adapter.bundle_size
        self.tst_slices = self.ntl_tst_slices + self.tvl_tst_slices

        # store visualization indices
        self.viz_inds = (self.ntl_tst_slices*viz_rand).astype(np.int32)

        self.training_pools = [{}, {}]
        for i in range(0, 2):
            self.training_pools[i]['ready'] = True
            self.training_pools[i]['trn_X'] = trn_X.copy()
            self.training_pools[i]['trn_Y'] = trn_Y.copy()

        self.serving_pool = 1
        # Set iterator up for loading next time
        self.slice_index = self.trn_slices


    def load_validation_set_and_first_training_pool(self, starting_j):
        """Load validation pool into memory

        Call this right before training is to start
        """
        self.val_X = np.zeros(
            (
                self.val_slices,
                self.adapter.height,
                self.adapter.width,
                self.adapter.in_channels
            ),
            np.float32
        )
        self.val_Y = np.zeros(
            (
                self.val_slices,
                self.adapter.height,
                self.adapter.width,
                1
            ),
            np.float32
        )
        (
            (
                self.val_X[0:self.ntl_val_slices,:,:,:],
                self.val_Y[0:self.ntl_val_slices,:,:,:]
            ),
            (
                self.val_X[self.ntl_val_slices:,:,:,:],
                self.val_Y[self.ntl_val_slices:,:,:,:]
            )
        ) = self.adapter.get_validation_pool()
        # set validation iterator up for starting at the beginning
        self.val_i = 0

        self.tst_X = np.zeros(
            (
                self.tst_slices,
                self.adapter.height,
                self.adapter.width,
                self.adapter.in_channels
            ),
            np.float32
        )
        self.tst_Y = np.zeros(
            (
                self.tst_slices,
                self.adapter.height,
                self.adapter.width,
                1
            ),
            np.float32
        )
        (
            (
                self.tst_X[0:self.ntl_tst_slices,:,:,:],
                self.tst_Y[0:self.ntl_tst_slices,:,:,:]
            ),
            (
                self.tst_X[self.ntl_tst_slices:,:,:,:],
                self.tst_Y[self.ntl_tst_slices:,:,:,:]
            )
        ) = self.adapter.get_testing_pool()
        # set validation iterator up for starting at the beginning
        self.tst_i = 0


        self.viz_X = self.tst_X[self.viz_inds,:,:,:]
        self.viz_Y = self.tst_Y[self.viz_inds,:,:,:]


        # Explicitly load first training pool
        self._new_training_pool(0, starting_j)


        # Instantiate thread to be joined
        self.worker = None


    def _new_training_pool(self, pool, j):
        """Load a new training pool into memory

        To be called when the previous pool has all been fed to the model for
        training.
        """
        print("_new_training_pool")
        bsz = self.adapter.bundle_size
        for i in range(0, self.trn_bundles):
            if self.type_order[j+i] > self.trivial_prob:
                ntl = True
            else:
                ntl = False
            (
                self.training_pools[pool]["trn_X"][i*bsz:(i+1)*bsz,:,:,:],
                self.training_pools[pool]["trn_Y"][i*bsz:(i+1)*bsz,:,:,:]
            ) = self.adapter.get_training_bundle(
                self.batch_order[j+i],
                ntl
            )
        self.training_pools[pool]["trn_X"] = (
            self.training_pools[pool]["trn_X"][self.pool_reorder,:,:,:]
        )
        self.training_pools[pool]["trn_Y"] = (
            self.training_pools[pool]["trn_Y"][self.pool_reorder,:,:,:]
        )
        self.training_pools[pool]["ready"] = True


    def start_new_worker(self, pool, j):
        other_pool = (pool + 1) % 2
        while not self.training_pools[other_pool]["ready"]:
            print("Waiting for pool %d to finish prepping" % other_pool)
            time.sleep(1)

        if self.worker is not None:
            self.worker.join()

        self.worker = threading.Thread(
            group=None,
            target=self._new_training_pool,
            name="worker",
            args=(pool, j),
            kwargs={}
        )
        self.worker.start()

    def next_training_batch(self, j, batch_size):
        """Check if need to load pool, load and shuffle if necessary

        This is where I might use multithreading/multiprocess, but
        experiments show me the benefits are limited
        Also may want to save any that may be left over if pool size is not
        a multiple of batch size.
        """
        self.trn_j = j

        while not self.training_pools[self.serving_pool]["ready"]:
            print("Waiting for pool %d" % self.serving_pool)
            time.sleep(5)

        if (self.slice_index + batch_size) >= self.trn_slices:
            print("Pool %d exhausted!" % self.serving_pool)
            self.training_pools[self.serving_pool]['ready'] = False
            old_pool = self.serving_pool
            self.serving_pool = (self.serving_pool + 1) % 2
            self.slice_index = 0
            self.start_new_worker(old_pool, j+self.trn_bundles)
            return self.next_training_batch(j, batch_size)

        i = self.slice_index
        self.slice_index = self.slice_index + batch_size

        return (
            self.training_pools[self.serving_pool]['trn_X'][i:i+batch_size,:,:,:],
            self.training_pools[self.serving_pool]['trn_Y'][i:i+batch_size,:,:,:]
        )

    def random_training_bundle(self):
        type_r = np.random.rand()
        ind_r = np.random.rand()
        if (type_r > self.trivial_prob):
            nontrivial = True
            print("Getting random training bundle: NONtrivial")
        else:
            nontrivial = False
            print("Getting random training bundle: trivial")

        return self.adapter.get_training_bundle(ind_r, nontrivial)


    def next_validation_batch(self, batch_size, prop_of_triv=1.0):
        """Returns next validation set in line

        Checks if next call would be legal (another full set beyond end of what
        was retured). Returns X, Y, True if so, X, Y, False if not and restarts
        """
        X = self.val_X[self.val_i*batch_size:(self.val_i+1)*batch_size]
        Y = self.val_Y[self.val_i*batch_size:(self.val_i+1)*batch_size]
        self.val_i = self.val_i + 1
        more_exists = True
        if (self.val_i+2)*batch_size > (self.ntl_val_slices + prop_of_triv*self.tvl_val_slices):
            more_exists = False
            self.val_i = 0

        return X, Y, more_exists


    def next_testing_batch(self, batch_size, prop_of_triv=1.0):
        """Returns next validation set in line

        Checks if next call would be legal (another full set beyond end of what
        was retured). Returns X, Y, True if so, X, Y, False if not and restarts
        """
        X = self.tst_X[self.tst_i*batch_size:(self.tst_i+1)*batch_size]
        Y = self.tst_Y[self.tst_i*batch_size:(self.tst_i+1)*batch_size]
        self.tst_i = self.tst_i + 1
        more_exists = True
        if (self.tst_i+2)*batch_size > (self.ntl_tst_slices + prop_of_triv*self.tvl_tst_slices):
            more_exists = False
            self.tst_i = 0

        return X, Y, more_exists


    def get_viz_batch(self):
        """Returns selected batch for visualization

        Batch is specified and stored in memory at construction time
        """
        return self.viz_X, self.viz_Y
