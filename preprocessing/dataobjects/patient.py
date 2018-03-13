class Patient(object):

    def __init__(self, pid, data, meta, gt=None):

        self.pid = pid
        self.data = data
        self.ground_truth = gt
        self.meta = meta
