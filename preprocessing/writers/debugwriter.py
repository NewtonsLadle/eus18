import numpy as np
from pathlib import Path
import scipy.misc as misc

class DebugWriter(object):

    def __init__(self, location):
        self.location = Path(location)

        if not self.location.exists():
            self.location.mkdir()

    def _write_to(self, loc, patient, info):
        volname = loc / (patient.pid + '-x')
        if not volname.exists():
            volname.mkdir()
        for i in range(0, patient.data.shape[0]):
            misc.imsave(str((volname / (str(i) + '.png'))), patient.data[i,:,:,0])
        infoname = volname / "info.npy"

        np.save(str(infoname), info)


    def write(self, patient, info):
        return self._write_to(self.location, patient, info)
