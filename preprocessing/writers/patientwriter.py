import numpy as np
from pathlib import Path

class PatientWriter(object):

    def __init__(self, location, trn_prop, val_prop):
        self.location = Path(location)
        self.trn_prop = trn_prop
        self.val_prop = val_prop

        self.training_dir = self.location / "training"
        self.validation_dir = self.location / "validation"
        self.testing_dir = self.location / "testing"

        if not self.location.exists():
            self.location.mkdir()
        if not self.training_dir.exists():
            self.training_dir.mkdir()
        if not self.validation_dir.exists():
            self.validation_dir.mkdir()
        if not self.testing_dir.exists():
            self.testing_dir.mkdir()

    def _write_to(self, loc, patient):
        volname = loc / (patient.pid + '-x.npy')
        segname = loc / (patient.pid + '-y.npy')
        np.save(str(volname), patient.data)
        np.save(str(segname), patient.ground_truth)


    def write(self, patient):
        r = np.random.rand()
        if r < self.trn_prop:
            self._write_to(self.training_dir, patient)
        elif r < (self.trn_prop+self.val_prop):
            self._write_to(self.validation_dir, patient)
        else:
            self._write_to(self.testing_dir, patient)
