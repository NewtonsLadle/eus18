import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from models.unet import UNet
from models.nullnet import NullNet
from models.threshnet import ThreshNet
from trainers.refinetrainer import RefineTrainer
from servers.medsegserver import MedSegServer
from configuration.configparser import ConfigParser
from testers.refinetester import RefineTester

from adapters.lisadapter import LiSAdapter
from adapters.psdpadapter import PSDPAdapter
from adapters.kidadapter import KidAdapter


"""
Produces organized prediction for a full patient
"""


CFGS_PATH = Path('/home/helle246/code/repos/urefine/configuration/cfgs')

MACHINE_CONFIG_FILE = CFGS_PATH / sys.argv[1]
SESSIONS_CONFIG_FILE = CFGS_PATH / sys.argv[2]

cfg = ConfigParser(MACHINE_CONFIG_FILE, SESSIONS_CONFIG_FILE)

s = cfg.s
m = cfg.m

with tf.device('/device:GPU:' + str(m['gpu'])):

    if (s['dataset'] == 'LiS'):
        print("Constructing LiS Adapter")
        adapter = LiSAdapter(
            lis_dir=m['lis_dir'],
            trn_lim=s['trn_lim']
        )
    elif (s['dataset'] == 'PSDP'):
        print("Constructing PSDP Adapter")
        adapter = PSDPAdapter(
            psdp_dir=m['psdp_dir'],
            trn_lim=s['trn_lim'],
            intensities=s['intensities'],
            locations=s['locations'],
            prefix=s['plane']
        )
    elif (s["dataset"] == 'Kid'):
        print("Constructing Kid Adapter")
        adapter = KidAdapter(
            sk_dir=m['sk_dir'],
            trn_lim=s['trn_lim'],
            intensities=s['intensities'],
            locations=s['locations'],
            out=s["leave_out"],
            plane=s["plane"]
        )
    else:
        print("Dataset:", s['dataset'], "not supported")
        sys.exit()

    print("Constructing server")
    server = MedSegServer(
        adapter,
        s['batch_order'],
        s['type_order'],
        s['viz_order'],
        s['trivial_prob'],
        s['training_bundles']
    )

    print("Creating feed-point placeholders")
    volume_shape = (
        None, adapter.height,
        adapter.width, adapter.in_channels
    )
    labels_shape = (
        None, adapter.height,
        adapter.width, 1
    )
    input_volume = tf.placeholder(
        tf.float32,
        shape=volume_shape,
        name='input_volume'
    )
    input_labels = tf.placeholder(
        tf.int64,
        shape=labels_shape,
        name='input_labels'
    )
    model_keep_prob = tf.placeholder_with_default(
        s['keep_prob'],
        (),
        name='model_keep_prob'
    )
    prior_keep_prob = tf.constant(1.0, name='prior_keep_prob')

    print("Constructing prior networks")
    nullprior = NullNet(input_volume, server, server.trivial_prob)
    #threshprior = ThreshNet(input_volume)
    priors = [nullprior] + [
        UNet(
            adapter.in_channels, adapter.out_channels, model['levels'],
            model['csize'], model['psize'], model['initial_filters'],
            prior_keep_prob, input_volume, False, None,
            model['primary_restore']['path'], model['secondary_restore']['path']
        )
        for model in s['priors']
    ]

    print("Constructing network to train")
    model = s['model']
    net = UNet(
        adapter.in_channels, adapter.out_channels, model['levels'],
        model['csize'], model['psize'], model['initial_filters'],
        model_keep_prob, input_volume, True, s['save_path'],
        model['primary_restore']['path'], model['secondary_restore']['path']
    )

    config = tf.ConfigProto(
        allow_soft_placement=True
    )
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:

        tester = RefineTester(server)

        print("Constructing trainer")
        trainer = RefineTrainer(
            input_volume,
            input_labels,
            model_keep_prob,
            net,
            priors,
            server,
            s['lr'],
            sess,
            s['tversky_alpha'],
            tester
        )

        print("Preparing server for training")
        server.load_validation_set_and_first_training_pool(s['starting_index'])

        print("Creating summaries")
        trainer.create_tensorboard_summaries(
            '/home/helle246/.tflogs',
            sess,
            s['log_name']
        )

        print("Initializing variables")
        sess.run(tf.global_variables_initializer())

        BS = 10

        X, Y = adapter.get_full_out()
        probs = np.zeros(Y.shape)

        num_slices = np.shape(X)[0]

        for i in range(0, num_slices//BS):
            start = i*BS
            stop = (i+1)*BS
            Xb = X[start:stop]
            Yb = Y[start:stop]

            probs[start:stop], f1 = sess.run(
                (trainer.val_refined_probs, trainer.val_f1),
                feed_dict={
                    input_volume: Xb,
                    input_labels: Yb,
                    net.keep_prob: 1.0
                }
            )

        print(probs.shape, str(model["primary_restore"]["path"]))
        np.save(str(model["primary_restore"]["path"] / "probs.npy"), probs)
