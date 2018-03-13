import json
import sys
import numpy as np
from pathlib import Path


def get_top_path(parent):
    """Return the path for the top performing set of weights in provided
    parent directory

    Exploits the fact that a new set of weights are only saved if the model
    got better.
    """
    print("Getting top path of " + str(parent))
    top_step = -1
    top_filename = ''
    for filename in parent.iterdir():
        step = int(filename.name[0:filename.name.find('_')])
        if step > top_step:
            top_step = step
            top_filename = filename

    print("Found" + str(top_filename))

    return top_filename

class ConfigParser(object):
    """The object which validates and holds dictionary configuration for a
    session

    Stores a session config dict along with a machinei config dict which are
    used lated when the program decides where to find things and what actions
    to perform
    """

    # Takes a string filename for machine configuration
    # Takes a list of string filenames for session configurations
    def __init__(self, machine_conf, session_conf, prior=False):
        """Construct the ConfigParser object

        Takes paths for machine configuration and session configuration,
        reads their contents as json objects and determines whether the
        provided values are legal
        """
        m_handle = machine_conf.open()
        s_handle = session_conf.open()
        machine_cfg = json.load(m_handle)
        self.m = machine_cfg
        session_cfg = json.load(s_handle)
        self.s = session_cfg

        self.prior = prior

        self._validate_machine_configuration()
        self._validate_session_configuration()


    def _validate_machine_configuration(self):
        """Check machine configuration file

        Does weights_dir exist? Kill if not
        Does source_dir exist? Kill if not
        Is gpu a positive integer? Kill if not
        """
        # Check weights dir
        self.m['weights_dir'] = Path(self.m['weights_dir'])
        if not self.m['weights_dir'].is_dir():
            print("ERROR: Invalid weights directory:", str(self.m['weights_dir']))
            sys.exit()
        # Check source dir
        self.m['source_dir'] = Path(self.m['source_dir'])
        if not self.m['source_dir'].is_dir():
            print("ERROR: Invalid source directory: ", str(self.m['source_dir']))
            sys.exit()
        # Check gpu
        if (not (isinstance(self.m['gpu'], (int))) and (self.m['gpu'] >= 0)):
            print("ERROR: Invalid gpu:", self.m['gpu'])
            sys.exit()


    def _get_top_path(self, parent):
        """Return the path for the top performing set of weights in provided
        parent directory

        Exploits the fact that a new set of weights are only saved if the model
        got better.
        """
        print("Getting top path of " + str(parent))
        top_step = -1
        top_filename = ''
        for filename in parent.iterdir():
            step = int(filename.name[0:filename.name.find('_')])
            if step > top_step:
                top_step = step
                top_filename = filename

        print("Found" + str(top_filename))

        return top_filename


    def _validate_restore(self, pat, typ):
        """Check the provided path to use for restoring a model

        If empty string provided, return None, else return a path
        object or kill if string provided cannot be resolved to valid dir
        """
        if pat != '':
            parent = self.m['weights_dir'] / pat
            if not parent.is_dir():
                print('ERROR: invalid restore directory:', str(parent))
                sys.exit()
            if (typ == 'top'):
                return self._get_top_path(parent)
            else:
                return parent
        else:
            return None

    def _validate_model(self, model):
        """Check restore paths for this model

        Returns model object with paths changed to Path object
        """
        # Check primary restore path, assigns it a Path object or None
        model['primary_restore']['path'] = self._validate_restore(
            model['primary_restore']['path'],
            model['primary_restore']['type']
        )
        # Check secondary restore path, assigns it a Path object or None
        model['secondary_restore']['path'] = self._validate_restore(
            model['secondary_restore']['path'],
            model['secondary_restore']['type']
        )
        return model


    def _validate_session_configuration(self):
        """Validate the provided session configuration file

        Validate each model
        Make sure save path can be created or exists and is empty, kill if not
        Make sure order file exists, kill if not
        """
        self.s['model'] = self._validate_model(self.s['model'])
        for i in range(0, len(self.s['priors'])):
            self.s['priors'][i] = self._validate_model(self.s['priors'][i])


        # Check save path
        self.s['log_name'] = self.s['save_path']
        self.s['save_path'] = self.m['weights_dir'] / self.s['save_path']
        if not self.prior:
            if self.s['save_path'].is_dir():
                if len([f for f in self.s['save_path'].iterdir()]) != 0:
                    print(
                        'ERROR: invalid save directory (exists and is not empty):',
                        str(self.s['save_path'])
                    )
                    sys.exit()
                else:
                    print(
                        'WARNING: save directory exists but is empty',
                        str(self.s['save_path'])
                    )
            else:
                self.s['save_path'].mkdir()

        # Check batch order file
        self.s['batch_order_file'] = self.m['source_dir'] / self.s['batch_order_file']
        if not self.s['batch_order_file'].exists():
            print('ERROR: invalid order file:', str(self.s['batch_order_file']))
            sys.exit()
        else:
            self.s['batch_order'] = np.load(str(self.s['batch_order_file']))

        # Check type order file
        self.s['type_order_file'] = self.m['source_dir'] / self.s['type_order_file']
        if not self.s['type_order_file'].exists():
            print('ERROR: invalid order file:', str(self.s['type_order_file']))
            sys.exit()
        else:
            self.s['type_order'] = np.load(str(self.s['type_order_file']))

        # Check viz order file
        self.s['viz_order_file'] = self.m['source_dir'] / self.s['viz_order_file']
        if not self.s['viz_order_file'].exists():
            print('ERROR: invalid order file:', str(self.s['viz_order_file']))
            sys.exit()
        else:
            self.s['viz_order'] = np.load(str(self.s['viz_order_file']))

        # Check trivial probability
        self.s['trivial_prob'] = float(self.s['trivial_prob'])
        if not ((self.s['trivial_prob'] >= 0) and (self.s['trivial_prob'] < 1)):
            print('ERROR: invalid trivial_prob:', self.s['trivial_prob'])
            sys.exit()
