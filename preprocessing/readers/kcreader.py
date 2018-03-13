from pathlib import Path
import numpy as np
import scipy.misc as misc

from dataobjects.patient import Patient


class KCReader(object):
    """A class to read annotations and slices from the kidney cancer dataset"""

    def __init__(self, annotations, volumes, mapping):
        """Construct the KC reader class

        Takes
            String location of annotation data (right from website)
            String location of raw volumes
        """
        self.annotations = Path(annotations)
        self.volumes = Path(volumes)
        self.mapping = mapping

        self.pairs = [
            (item, mapping[item.name])
            for item in self.annotations.iterdir()
        ]

        self.num_patients = len(self.pairs)

        self.i = 0


    def get_seg(self, k, annotations):
        lseg = None
        rseg = None
        lmax = 0
        rmax = 0
        for annotation in annotations:
            parts = annotation.stem.split('_')
            n = int(parts[-1])
            if (n == k):
                if (parts[2] == "left") and (parts[3] == 'kidney'):
                    if (int(parts[0]) > lmax):
                        lmax = int(parts[0])
                        lseg = annotation
                if (parts[2] == "right") and (parts[3] == 'kidney'):
                    if (int(parts[0]) > rmax):
                        rmax = int(parts[0])
                        rseg = annotation


        return lseg, rseg


    def prepreprocess(self, lseg, rseg, warn):
        ret = np.zeros((512, 512, 1))
        if (lseg is None) and (rseg is None):
            print("WARNING - empty but shouldn't be", warn)
            return np.zeros((512, 512, 1))
        if (lseg is not None):
            ret = ret + np.reshape(misc.imresize(misc.imread(str(lseg), mode='L'), (512,512)), (512,512,1))
        if (rseg is not None):
            ret = ret + np.reshape(misc.imresize(misc.imread(str(rseg), mode='L'), (512,512)), (512, 512, 1))
        return ret


    def next_patient(self):
        if self.i >= self.num_patients:
            return None

        (gt_dir, info) = self.pairs[self.i]

        ann_dir = gt_dir / "annotated"

        annotations = [
            item for item in ann_dir.iterdir()
        ]


        volume = self.volumes / (info["volume"] + '.npy')
        vol = np.load(str(volume))

        j = 0
        for i in range(0, vol.shape[0]):
            if ((i - info['start']) % info['skip']) == 0:
                j = j + 1

        retvol = np.zeros((j, 512, 512, 1))
        retseg = np.zeros((j, 512, 512, 1))

        print("Checking %s:" % gt_dir.name)

        j = 0
        k = 0
        for i in range(0, vol.shape[0]):
            if ((i - info['start']) % info['skip']) == 0:
                retvol[j] = vol[i]
                if ((i >= info['start']) and (k < info['num'])):
                    lseg, rseg = self.get_seg(k, annotations)
                    retseg[j] = self.prepreprocess(lseg, rseg, (i,j,k))
                    k = k + 1
                j = j + 1




        self.i = self.i + 1

        patient = Patient(gt_dir.name, retvol, {}, retseg)
        patient.include_outline = info["include_outline"]
        return patient, self.i < self.num_patients
