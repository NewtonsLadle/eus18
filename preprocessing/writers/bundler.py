import numpy as np
from pathlib import Path

class Bundler(object):

    def __init__(self, bundle_size, height, width,
                 intensity_channels,
                 location_channels):
        self.bundle_size = bundle_size
        self.height = height
        self.width = width
        self.channels = intensity_channels + location_channels
        self.intensity_channels = intensity_channels
        self.location_channels = location_channels


    def save(self, X, Y, j, typ, loc, pfx):
        print(j, typ)
        np.save(
            str(loc / (pfx+'-X-'+typ+'-'+str(j)+'.npy')),
            X[:,:,:,0:self.intensity_channels]
        )
        if self.location_channels > 0:
            np.save(
                str(loc / (pfx+'-L-'+typ+'-'+str(j)+'.npy')),
                X[:,:,:,self.intensity_channels:]
            )
        np.save(
            str(loc / (pfx+'-Y-'+typ+'-'+str(j)+'.npy')),
            Y
        )


    def add(self, i, j, X, Y, img, seg, loc, typ, pfx):
        X[i] = img
        Y[i] = seg
        i = i + 1

        if (i >= self.bundle_size):
            self.save(X, Y, j, typ, loc, pfx)
            i = 0
            j = j + 1
        return i, j, X, Y



    def bundle(self, loc, prefix):
        self.ntlX = np.zeros(
            (self.bundle_size, self.height, self.width, self.channels)
        )
        self.ntlY = np.zeros(
            (self.bundle_size, self.height, self.width, 1)
        )
        self.tvlX = np.zeros(
            (self.bundle_size, self.height, self.width, self.channels)
        )
        self.tvlY = np.zeros(
            (self.bundle_size, self.height, self.width, 1)
        )
        self.ntl_i = 0
        self.tvl_i = 0
        self.ntl_j = 0
        self.tvl_j = 0


        for x in loc.glob(prefix + '*-x.npy'):
            ind = int(x.name[-10:-6])
            for y in loc.glob(prefix + '*%04d-y.npy' % ind):
                break
            patient_vol = np.load(str(x))
            patient_seg = np.load(str(y))
            print(x.name)
            print(y.name)
            print()
            for i in range(0, patient_vol.shape[0]):
                decider = np.sum(np.equal(patient_seg[i], 1).astype(np.float32))
                if decider > 0.5:
                    (
                        self.ntl_i, self.ntl_j, self.ntlX, self.ntlY
                    ) = self.add(
                        self.ntl_i, self.ntl_j, self.ntlX, self.ntlY,
                        patient_vol[i], patient_seg[i], loc, 'ntl', prefix
                    )
                else:
                    (
                        self.tvl_i, self.tvl_j, self.tvlX, self.tvlY
                    ) = self.add(
                        self.tvl_i, self.tvl_j, self.tvlX, self.tvlY,
                        patient_vol[i], patient_seg[i], loc, 'tvl', prefix
                    )

        self.save(self.ntlX, self.ntlY, self.ntl_j, 'ntl', loc, prefix)
        self.save(self.tvlX, self.tvlY, self.tvl_j, 'tvl', loc, prefix)
