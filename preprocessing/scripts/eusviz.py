import sys
import numpy as np
import scipy.misc as misc
import json
from pathlib import Path

from visualizers.viztriple import VizTriple
from readers.kcreader import KCReader
from processors.kidneythresholder import KidneyThresholder
from processors.hilumfiller import HilumFiller


if __name__ == "__main__":

    vt = VizTriple(-512, 512)

    predictions_dir = Path("/home/helle246/data/urefine/predictions")

    # Read configuration file
    cfg_file = Path("/home/helle246/code/repos/medicalpreprocessing/configuration/kidneys.json")
    cfg_handle = cfg_file.open()
    cfg = json.load(cfg_handle)

    # Read slice number and set name from command line
    ind = int(sys.argv[1])
    set_name = sys.argv[2]

    # Build reader
    reader = KCReader(cfg['annotations'], cfg['volumes'], cfg["mapping"])

    # Build processors
    thresholder = KidneyThresholder(10.0)
    filler = HilumFiller()


    # Iterate over patients
    more_patients = True
    while more_patients:
        patient, more_patients = reader.next_patient()
        print(patient.pid, patient.data.shape, patient.ground_truth.shape)

        if (patient.pid == set_name):

            Y = patient.ground_truth

            mini = np.shape(Y)[0]
            for i in range(0, Y.shape[0]):
                if np.sum(Y[i]) > 0.5:
                    if i < mini:
                        mini = i


            predictions = np.load(str(predictions_dir / ("%s.npy" % patient.pid)))

            outline = patient.ground_truth[ind]

            computed_ground_truth = thresholder.apply_threshold(
                patient.ground_truth[ind,:,:,0],
                patient.data[ind,:,:,0],
                patient.include_outline
            )

            computed_ground_truth[:,:256], modified = filler.fill(
                computed_ground_truth[:,:256]
            )
            computed_ground_truth[:,256:], modified = filler.fill(
                computed_ground_truth[:,256:]
            )

            # Apply threshold again
            computed_ground_truth[:,:] = thresholder.apply_threshold(
                computed_ground_truth[:,:],
                patient.data[ind,:,:,0],
                True
            )

            cstart = cfg["mapping"][patient.pid]["start"]
            cskip = cfg["mapping"][patient.pid]["skip"]

            pred_ind = ind - (mini - 23) - 1

            vt.create_triple(
                Path("./"),
                patient.data[ind,:,:,0],
                outline,
                computed_ground_truth,
                predictions[pred_ind,:,:,0]
            )


            break
