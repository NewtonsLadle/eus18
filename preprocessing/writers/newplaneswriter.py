import numpy as np
from pathlib import Path

class NewPlanesWriter(object):

    def __init__(self, location):
        self.location = Path(location)

        self.axial_dir = self.location / "axial"
        self.sagittal_dir = self.location / "sagittal"
        self.coronal_dir = self.location / "coronal"

        if not self.location.exists():
            self.location.mkdir()
        if not self.axial_dir.exists():
            self.axial_dir.mkdir()
        if not self.sagittal_dir.exists():
            self.sagittal_dir.mkdir()
        if not self.coronal_dir.exists():
            self.coronal_dir.mkdir()

    def _write_to(self, loc, patient, data, ground_truth):
        volname = loc / (patient.pid + '-x.npy')
        segname = loc / (patient.pid + '-y.npy')
        np.save(str(volname), data)
        np.save(str(segname), ground_truth)


    def write(self, patient):
        self._write_to(self.axial_dir, patient, patient.axial, patient.axial_gt)
        self._write_to(self.sagittal_dir, patient, patient.sagittal, patient.sagittal_gt)
        self._write_to(self.coronal_dir, patient, patient.coronal, patient.coronal_gt)
