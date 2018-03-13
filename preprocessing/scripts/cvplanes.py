import numpy as np
import json
from pathlib import Path

if __name__ == "__main__":

    # Read configuration file
    cfg_file = Path("/home/helle246/code/repos/medicalpreprocessing/configuration/kidneys.json")
    cfg_handle = cfg_file.open()
    cfg = json.load(cfg_handle)

    location = Path(cfg["write_location"])

    # Iterate over files
    full_files = location.glob("*-x.npy")
    for full_file in full_files:
        name = full_file.stem[:-2]
        volume_file = location / (name + '-x.npy')
        segmentation_file = location / (name + '-y.npy')
        X = np.load(str(volume_file))
        Y = np.load(str(segmentation_file))
        mini = Y.shape[0]
        maxi = -1
        for i in range(0, Y.shape[0]):
            if np.sum(Y[i]) > 0.5:
                if i < mini:
                    mini = i
                if i > maxi:
                    maxi = i
        print(name, mini, maxi)
        ax_X = X[mini-23:77+mini]
        ax_Y = Y[mini-23:77+mini]
        np.save(str(location / ("axial-" + name + '-x.npy')), ax_X)
        np.save(str(location / ("axial-" + name + '-y.npy')), ax_Y)

        sa_X = np.transpose(ax_X, [2,1,0,3])
        sa_Y = np.transpose(ax_Y, [2,1,0,3])
        np.save(str(location / ("sagittal-" + name + '-x.npy')), sa_X)
        np.save(str(location / ("sagittal-" + name + '-y.npy')), sa_Y)

        co_X = np.transpose(ax_X, [1,0,2,3])
        co_Y = np.transpose(ax_Y, [1,0,2,3])
        np.save(str(location / ("coronal-" + name + '-x.npy')), co_X)
        np.save(str(location / ("coronal-" + name + '-y.npy')), co_Y)
