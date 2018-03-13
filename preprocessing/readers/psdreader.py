import numpy as np
from pathlib import Path
import dicom
import nibabel as nib

from dataobjects.patient import Patient


class PSDReader(object):

    def __init__(self, volume_loc, labels_loc):
        self.volume_path = Path(volume_loc)
        self.labels_path = Path(labels_loc)

        self.num_patients = 82
        self.height = 512
        self.width = 512
        self.channels = 1

        self.slim_patients = []

        for i in range(1, self.num_patients+1):
            pid = "PANCREAS_%04d" % i
            volume_subdir = self.get_subdir(i)
            labels_file = self.get_gt_file(i)
            self.slim_patients = self.slim_patients + [(
                volume_subdir,
                len([x for x in volume_subdir.iterdir()]),
                labels_file,
                pid
            )]

        self.i = 0

    def get_gt_file(self, i):
        return self.labels_path / ("label%04d.nii.gz" % i)

    def get_subdir(self, i):
        root = self.volume_path / ("PANCREAS_%04d" % i)
        for d in root.iterdir():
            for e in d.iterdir():
                return e

    def next_patient(self):
        (path, num, gt, pid) = self.slim_patients[self.i]

        positions = np.zeros(num)


        data = np.zeros((num, self.height, self.width, self.channels), np.float32)
        for j in range(0, num):
            slice_name = path / ("%06d.dcm" % j)
            dcm = dicom.read_file(str(slice_name))
            instance_num = int(dcm.InstanceNumber)
            positions[instance_num-1] = float(dcm[0x20,0x32][2])

            data[instance_num-1] = np.reshape(
                np.transpose(dcm.pixel_array),
                [512,512,1]
            )

        meta = {}
        meta["positions"] = positions
        meta["spacing"] = np.array(dcm[0x28,0x30].value).astype(np.float32)

        gt = np.reshape(
            np.transpose(
                nib.load(str(gt)).get_data(),
                [2,0,1]
            ),
            (num, self.height, self.width, 1)
        )


        patient = Patient(pid, data, meta, gt)
        pi = int(pid[-4:])
        patient.pi = pi
        self.i = self.i + 1

        return patient, ((self.i) < 82)
