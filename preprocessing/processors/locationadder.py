import numpy as np

class LocationAdder(object):

    def __init__(self, height, width):
        atom = np.transpose(np.arange(-1.0+1.0/512.0, 1.0, 1.0/256.0))
        self.coronal_slice = np.array(
            [atom for i in range(0, 512)],
            np.float32
        )
        self.sagittal_slice = np.transpose(self.coronal_slice.copy())
        self.transverse_slice = np.ones((512,512), np.float32)

        self.coronal_slice = np.reshape(self.coronal_slice, (512, 512, 1))
        self.sagittal_slice = np.reshape(self.sagittal_slice, (512, 512, 1))
        self.transverse_slice = np.reshape(self.transverse_slice, (512, 512, 1))


    def add_locations(self, patient):
        batch_size = patient.data.shape[0]
        coronal = np.stack(
            [self.coronal_slice for i in range(0, batch_size)]
        )
        sagittal = np.stack(
            [self.sagittal_slice for i in range(0, batch_size)]
        )
        transverse = np.stack(
            [
                (
                    (2.0*(i-batch_size/2.0 + 2.0/batch_size)/(batch_size))
                    *self.transverse_slice
                )
                for i in range(0, batch_size)
            ]
        )
        patient.data = np.concatenate(
            (
                patient.data,
                coronal,
                sagittal,
                transverse
            ),
            axis=3
        )
        return patient
