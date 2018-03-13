from pathlib import Path
import json
import sys
import numpy as np

"""
Takes a meta configuration file and produces configuration files for all
appropriate training runs, along with shell scripts to drive them
"""


anatomical_planes = ["axial", "sagittal", "coronal"]


cfg_target = "generated2"
cfgs_path = Path('./configuration')
put_path = cfgs_path / "cfgs" / cfg_target
if not put_path.exists():
    put_path.mkdir()
script_base = Path('./scripts') / cfg_target
if not script_base.exists():
    script_base.mkdir()




def get_name(i, out, plane):
    return out + "_%s_%d" % (plane, i+1)

def get_model(meta, levels, primary_restore=None):
    ret = {}
    if primary_restore is None:
        primary_restore = meta["default_weights"]
        primary_restore_type = "exact"
    else:
        primary_restore_type = "top"
    ret["primary_restore"] = {
        "path": primary_restore,
        "type": primary_restore_type
    }
    ret["secondary_restore"] = {
        "path": "",
        "type": ""
    }
    ret["levels"] = levels
    ret["csize"] = 3
    ret["psize"] = 2
    ret["initial_filters"] = 10
    return ret

def get_train(meta, out, plane, i, restore=None):
    ret = {}
    ret["model"] = get_model(meta, meta["levels"][i], restore)
    priors = []
    for j in np.arange(i-1,-1,-1):
        priors = [
            get_model(meta, meta["levels"][j], get_name(j, out, plane))
        ] + priors
    ret["priors"] = priors
    ret["tversky_alpha"] = meta["alphas"][i]
    if restore is not None:
        ret["save_path"] = "test"
    else:
        ret["save_path"] = get_name(i, out, plane)
    ret["batch_order_file"] = meta["batch_order_file"]
    ret["type_order_file"] = meta["type_order_file"]
    ret["viz_order_file"] = meta["viz_order_file"]
    ret["trivial_prob"] = meta["trivial_probs"][plane][i]
    index = 0
    for j in np.arange(i-1, -1, -1):
        index = index + meta["epochs"][j]
    ret["starting_index"] = index
    ret["epochs"] = meta["epochs"][i]
    ret["save_threshold"] = 0.0
    ret["display_step"] = meta["display_steps"][i]
    ret["batch_size"] = meta["batch_sizes"][plane]
    ret["lr"] = meta["lrs"][i]
    ret["dataset"] = meta["dataset"]
    ret["plane"] = plane
    ret["trn_lim"] = -1
    ret["keep_prob"] = 1.0
    ret["training_bundles"] = meta["training_bundles"][plane]
    ret["intensities"] = True
    ret["locations"] = False
    ret["leave_out"] = out
    return ret


def get_prior(meta, out, plane, i):
    ret = {}
    ret["model"] = get_model(meta, meta["levels"][i-1], get_name(i-1, out, plane))
    priors = []
    for j in np.arange(i-2,-1,-1):
        priors = [
            get_model(meta, meta["levels"][j], get_name(j, out, plane))
        ] + priors
    ret["priors"] = priors
    ret["tversky_alpha"] = meta["alphas"][i]
    ret["save_path"] = "prior%d" % i
    ret["batch_order_file"] = meta["batch_order_file"]
    ret["type_order_file"] = meta["type_order_file"]
    ret["viz_order_file"] = meta["viz_order_file"]
    ret["trivial_prob"] = meta["trivial_probs"][plane][i]
    index = 0
    for j in np.arange(i-1, -1,-1):
        index = index + meta["epochs"][j]
    ret["starting_index"] = index
    ret["epochs"] = meta["epochs"][i]
    ret["save_threshold"] = 0.0
    ret["display_step"] = meta["display_steps"][i]
    ret["batch_size"] = meta["batch_sizes"][plane]
    ret["lr"] = meta["lrs"][i]
    ret["dataset"] = meta["dataset"]
    ret["plane"] = plane
    ret["trn_lim"] = -1
    ret["keep_prob"] = 1.0
    ret["training_bundles"] = meta["training_bundles"][plane]
    ret["intensities"] = True
    ret["locations"] = False
    ret["leave_out"] = out
    return ret


def get_training_sessions(meta, out):
    ret = []
    for plane in anatomical_planes:
        num_training_sessions = len(meta["levels"])
        # Make first training session
        training_sessions = [get_train(meta, out, plane, 0)]
        prior_sessions = []
        for i in range(1, num_training_sessions):
            # Make prior prep session
            prior_sessions = prior_sessions + [
                get_prior(meta, out, plane, i)
            ]
            # Make training session
            training_sessions = training_sessions + [
                get_train(meta, out, plane, i)
            ]

        prior_sessions = prior_sessions + [
            get_train(meta, out, plane, num_training_sessions-1, get_name(num_training_sessions-1, out, plane))
        ]


        ret = ret + [(training_sessions, prior_sessions)]
    return ret


def write_files(pth, config, machine, out, plane, mts, mts_mach):
    script_file = script_base / (out + '_' + plane + '.sh')
    with script_file.open('w') as sf:
        for i in range(0, 2*len(config[0]) - 1):
            if (i % 2 == 0):
                fl = pth / ("train%d.json" % ((i+2)//2))
                with fl.open('w') as f:
                    json.dump(config[0][i//2], f)
                sf.write(
                    'python3 -m scripts.run machines/%s.json %s/%s/%s/train%d.json\n' % (
                            machine, cfg_target, out, plane, (i+2)//2
                    )
                )
            else:
                fl = pth / ("prior%d.json" % ((i+1)//2))
                with fl.open('w') as f:
                    json.dump(config[1][i//2], f)
                sf.write(
                    'python3 -m scripts.prepare_prior machines/%s.json %s/%s/%s/prior%d.json\n' % (
                            machine, cfg_target, out, plane, (i+1)//2
                    )
                )
            fl = pth / ("test.json")
            with fl.open('w') as f:
                json.dump(config[1][(2*len(config[0]) - 1)//2], f)

    mts.write(
        'python3 -m scripts.predict_patient machines/%s %s/%s/%s/test.json\n'%(
            mts_mach, cfg_target, out, plane
        )
    )





if __name__ == "__main__":
    meta = json.load((cfgs_path / sys.argv[1]).open())

    machines = [["jupiter1", "jupiter2", "jupiter3"],
                ["jinx0", "jinx1", "jinx2"]]

    i = 0
    master_test_script = script_base / ('master_test.sh')
    mts_mach = "jupiter1.json"
    with master_test_script.open('w') as mts:
        for out in meta["outs"]:
            s = get_training_sessions(meta, out)
            i = i + 1

            cfg_base = put_path / out
            if not cfg_base.exists():
                cfg_base.mkdir()
            j = 0
            for plane in anatomical_planes:
                cfg_folder = cfg_base / plane
                if not cfg_folder.exists():
                    cfg_folder.mkdir()
                # Write configuration files and script
                machine = machines[i % len(machines)][j]
                write_files(cfg_folder, s[j], machine, out, plane, mts, mts_mach)
                j = j + 1
