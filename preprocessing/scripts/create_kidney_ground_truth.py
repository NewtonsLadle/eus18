import json
from pathlib import Path
import numpy as npy

from readers.kcreader import KCReader
from writers.patientwriter import PatientWriter
from writers.kidneybundler import KidneyBundler
from processors.kidneythresholder import KidneyThresholder
from processors.hilumfiller import HilumFiller

if __name__ == "__main__":


    # Read configuration file
    cfg_file = Path("/home/helle246/code/repos/medicalpreprocessing/configuration/kidneys2.json")
    cfg_handle = cfg_file.open()
    cfg = json.load(cfg_handle)

    # Build reader
    reader = KCReader(cfg['annotations'], cfg['volumes'], cfg["mapping"])

    # Build processors
    thresholder = KidneyThresholder(10.0)
    filler = HilumFiller()

    # Build writer
    writer = PatientWriter(cfg['write_location'], 1.0, 0.0)

    # Iterate over patients
    more_patients = True
    while more_patients:
        patient, more_patients = reader.next_patient()
        print(patient.pid, patient.data.shape, patient.ground_truth.shape)

        # Set all gt marks to 1 or 0
        patient.ground_truth[patient.ground_truth > 0.5] = 1

        #Iterate over all slices
        for i in range(0, patient.data.shape[0]):

            # Apply threshold and containment
            patient.ground_truth[i,:,:,0] = thresholder.apply_threshold(
                patient.ground_truth[i,:,:,0],
                patient.data[i,:,:,0],
                patient.include_outline
            )

            # Apply line cropping
            patient.ground_truth[i,:,:256,0], modified = filler.fill(
                patient.ground_truth[i,:,:256,0]
            )
            patient.ground_truth[i,:,256:,0], modified = filler.fill(
                patient.ground_truth[i,:,256:,0]
            )

            # Apply threshold again
            patient.ground_truth[i,:,:,0] = thresholder.apply_threshold(
                patient.ground_truth[i,:,:,0],
                patient.data[i,:,:,0],
                True
            )


        # Save slices and ground truth to disk
        writer.write(patient)


    """
    Stop the code here and run cvplanes, then come back (I think there's a directory issue too...)
    Sorry about this, never got around to cleaning it up
    """

    # Build bundler
    bundler = KidneyBundler(20, 512, 512, 1, 0)

    bundler.bundle(writer.training_dir, "axial-blake367")
    bundler.bundle(writer.training_dir, "axial-cjweight")
    bundler.bundle(writer.training_dir, "axial-ravis071")
    bundler.bundle(writer.training_dir, "axial-srkrishn")
    bundler.bundle(writer.training_dir, "axial-trade011")
    bundler.bundle(writer.training_dir, "axial-nsathian")

    bundler = KidneyBundler(20, 512, 100, 1, 0)

    bundler.bundle(writer.training_dir, "sagittal-blake367")
    bundler.bundle(writer.training_dir, "sagittal-cjweight")
    bundler.bundle(writer.training_dir, "sagittal-ravis071")
    bundler.bundle(writer.training_dir, "sagittal-srkrishn")
    bundler.bundle(writer.training_dir, "sagittal-trade011")
    bundler.bundle(writer.training_dir, "sagittal-nsathian")

    bundler = KidneyBundler(20, 100, 512, 1, 0)

    bundler.bundle(writer.training_dir, "coronal-blake367")
    bundler.bundle(writer.training_dir, "coronal-cjweight")
    bundler.bundle(writer.training_dir, "coronal-ravis071")
    bundler.bundle(writer.training_dir, "coronal-srkrishn")
    bundler.bundle(writer.training_dir, "coronal-trade011")
    bundler.bundle(writer.training_dir, "coronal-nsathian")
