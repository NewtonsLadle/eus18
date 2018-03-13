import numpy as np
from dataobjects.patient import Patient

class PancreasFilter(object):

    def __init__(self):
        self.floor = -5.0
        self.ceiling = 175.0
        self.range = self.ceiling - self.floor

    def filter(self, patient):
        tmp_dat = patient.data

        tmp_dat[tmp_dat<self.floor] = self.floor
        tmp_dat[tmp_dat>self.ceiling] = self.ceiling
        tmp_dat = 2.0*(tmp_dat-self.floor-(self.range/2))/(self.range)

        patient.data = tmp_dat
        return patient
