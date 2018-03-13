from pathlib import Path
import numpy as np

def _get_dict(dict_path):
    """Get dictionary of variables stored at given path

    Name stems are set as keys
    """
    dicti = {}
    path = dict_path
    if (path is not None):
        wt_fls = [f for f in path.iterdir() if f.is_file()]
        for f in wt_fls:
            name = f
            key = f.stem
            dicti[key] = np.load(name)
    return dicti


class NetManager(object):
    """The operating system interface for model weights

    Belongs to a model. Used for restoring it before training and saving
    it after or during training.
    """

    def __init__(self, primary_restore_path, secondary_restore_path,
                 save_root):
        """Constructor for netmanager object

        Takes three path objects:
            primary_restore
            secondary_restore
            and root_save_path
        These have been validated already by the config parser
        """
        self.primary_restore_dict = _get_dict(primary_restore_path)
        self.secondary_restore_dict = _get_dict(secondary_restore_path)
        self.save_root = save_root


    def _get_save_path(self, j, measure):
        """Returns the path to save the weights in this time

        is based on what step it is and how well it performed
        must not currently exist, will make it
        """
        new_dir = str(j) + '_' + str(measure)
        ret = self.save_root / new_dir
        ret.mkdir()
        return ret


    def save_model(self, vardict, meta_dict, j, measure):
        """Save the model weights to disk

        Gets location from internal call and iterates of dict of variables
        """
        location = self._get_save_path(j, measure)
        for key, val in vardict.items():
            name = location / (key + '.npy')
            np.save(str(name), val)
        for key, val in meta_dict.items():
            name = location / (key + '.npy')
            np.save(str(name), val)
