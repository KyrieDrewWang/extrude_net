import os
import glob
import open3d as o3d
import numpy as np
import argparse
from joblib import Parallel, delayed
import random
from scipy.spatial import cKDTree as KDTree
import time
import sys
sys.path.append("..")
sys.path.append(".")
from plyfile import PlyData, PlyElement

def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex
# data that is unable to process
SKIP_DATA = [""]


def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points


def process_one(path):

    data_id = path.split('/')[-1].split('.')[0]
    
    gt_pc_path = os.path.join(PC_ROOT, data_id[:4], data_id + '_surface_point_cloud.ply')
    
    if not os.path.exists(gt_pc_path):
        return None

    # if np.max(np.abs(out_pc)) > 2: # normalize out-of-bound data
    #     out_pc = normalize_pc(out_pc)

    # out_pc = read_ply(path)
    # out_pc = out_pc - np.mean(out_pc, axis=0)

    pred_mesh = o3d.io.read_triangle_mesh(path)

    out_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(pred_mesh, number_of_points=args.n_points)
    out_pc = np.array(out_pc.points)
    out_pc = out_pc - np.mean(out_pc)    

    gt_pc = read_ply(gt_pc_path)
    gt_pc = gt_pc - np.mean(gt_pc, axis=0)
    sample_idx = random.sample(list(range(gt_pc.shape[0])), args.n_points)
    gt_pc = gt_pc[sample_idx]

    cd = chamfer_dist(gt_pc, out_pc)
    return cd


def run(args):
    filepaths = sorted(glob.glob(os.path.join(args.src, "*.ply")))
    if args.num != -1:
        filepaths = filepaths[:args.num]

    save_path = args.src + '_pc_stat.txt'
    record_res = None
    if os.path.exists(save_path):
        response = input(save_path + ' already exists, overwrite? (y/n) ')
        if response == 'y':
            os.system("rm {}".format(save_path))
            record_res = None
        else:
            with open(save_path, 'r') as fp:
                record_res = fp.readlines()
                n_processed = len(record_res) - 3

    if args.parallel:
        dists = Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in filepaths)
    else:
        dists = []
        for i in range(len(filepaths)):
            print("processing[{}] {}".format(i, filepaths[i]))
            data_id = filepaths[i].split('/')[-1].split('.')[0]

            if record_res is not None and i < n_processed:
                record_dist = record_res[i].split('\t')[-1][:-1]
                record_dist = None if record_dist == 'None' else eval(record_dist)
                dists.append(record_dist)
                continue

            if data_id in SKIP_DATA:
                print("skip {}".format(data_id))
                res = None
            else:
                res = process_one(filepaths[i])
                # if res >= 1.0:
                #     continue
            with open(save_path, 'a') as fp:
                print("{}\t{}\t{}".format(i, data_id, res), file=fp)
            dists.append(res)

    valid_dists = [x for x in dists if x is not None]
    valid_dists = sorted(valid_dists)
    print("top 20 largest error:")
    print(valid_dists[-20:][::-1])
    n_valid = len(valid_dists)
    n_invalid = len(dists) - n_valid

    avg_dist = np.mean(valid_dists)
    trim_avg_dist = np.mean(valid_dists[int(n_valid * 0.1):-int(n_valid * 0.1)])
    med_dist = np.median(valid_dists)

    print("#####" * 10)
    print("total:", len(filepaths), "\t invalid:", n_invalid, "\t invalid ratio:", n_invalid / len(filepaths))
    print("avg dist:", avg_dist, "trim_avg_dist:", trim_avg_dist, "med dist:", med_dist)
    with open(save_path, "a") as fp:
        print("#####" * 10, file=fp)
        print("total:", len(filepaths), "\t invalid:", n_invalid, "\t invalid ratio:", n_invalid / len(filepaths),
              file=fp)
        print("avg dist:", avg_dist, "trim_avg_dist:", trim_avg_dist, "med dist:", med_dist,
              file=fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default="/data/wc/extrude_net/samples/pc2skh1", required=False)
    parser.add_argument('--n_points', type=int, default=2000)
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--parallel', action='store_true', help="use parallelization")
    parser.add_argument('--pc_root', type=str, default="/data/wc/Points2sketch/DATABASE/deepcad/pc2skh_3d/mesh/", required=False)
    args = parser.parse_args()
    PC_ROOT = args.pc_root
    print(args.src)
    print("SKIP DATA:", SKIP_DATA)
    since = time.time()
    run(args)
    end = time.time()
    print("running time: {}s".format(end - since))