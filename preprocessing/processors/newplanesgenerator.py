import numpy as np
from dataobjects.patient import Patient

class NewPlanesGenerator(object):

    def __init__(self):
        pass

    def generate_planes(self, patient, height, width):

        num_slices = patient.data.shape[0]
        max_num_slices = 466

        patient.axial = patient.data.copy()
        patient.sagittal = np.zeros((width, max_num_slices, height, 1), np.float32)
        patient.coronal = np.zeros((height, max_num_slices, width, 1), np.float32)

        patient.axial_gt = patient.ground_truth.copy()
        patient.sagittal_gt = 2*np.ones((width,  max_num_slices, height, 1), np.float32)
        patient.coronal_gt = 2*np.ones((height,  max_num_slices, width, 1), np.float32)

        offset = (max_num_slices-num_slices)//2
        miraxial = np.zeros((max_num_slices, height, width, 1), np.float32)
        for i in range(0, max_num_slices):
            if (i < offset):
                ind = offset - i
            elif (i >= (offset + num_slices)):
                ind = offset + num_slices - i - 1
            else:
                ind = i - offset
            miraxial[i,:,:,:] = patient.axial[ind,:,:,:]

        for i in range(0, width):
            patient.sagittal[i] = miraxial[:,i,:,:]
        for i in range(0, height):
            patient.coronal[i] = miraxial[:,:,i,:]

        for i in range(0, width):
            patient.sagittal_gt[i,offset:offset+num_slices,:,:] = patient.axial_gt[:,i,:,:]
        for i in range(0, height):
            patient.coronal_gt[i,offset:offset+num_slices,:,:] = patient.axial_gt[:,:,i,:]

        return patient
