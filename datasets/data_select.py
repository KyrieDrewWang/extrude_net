import os
import glob
import shutil
import json

CACHE_PATH = "/data/wc/Points2sketch/DATABASE/pc2skh_3d/data.txt"
DATASET_PATH = "/data/wc/Points2sketch/DATABASE/pc2skh_3d/mesh"

def dict2json(the_dict, file_name):
    json_str = json.dumps(the_dict)
    with open(file_name, 'w') as json_file:
        json_file.write(json_str)

def pathrename(obj_path, suffix):
    '''
    given the absolute path of the file, return the file name without suffix
    '''
    base_path, file_name = os.path.split(obj_path)
    file_id = file_name.split('.')[0]
    new_path = os.path.join(base_path, file_id + suffix)
    return new_path

def get_all_obj_path(use_cache=True):
    # Using cached obj file paths
    # Scan directory for all obj files
    if use_cache:
        files = []
        assert os.path.exists(CACHE_PATH), "Please make sure the cache file %s is avaliable..." % CACHE_PATH
        print("Using cached paths to retrive all obj files...")
        with open(CACHE_PATH, "r") as f:
            lines = f.readlines()
        for l in lines:
            if l != '\n':
                files.append(l.strip())
    else:
        files = []
        print("Gathering all obj files...")
        files = glob.glob("%s/**/*.obj" % DATASET_PATH, recursive=True)
        with open(CACHE_PATH, "w") as f:
            for path in files:
                f.write(path + "\n")
    return files

if __name__ == "__main__":
    files = get_all_obj_path(use_cache=True)
    s_dir = "/data/wc/extrude_net/data/pc2skh/surface_pc"
    o_dir = "/data/wc/extrude_net/data/pc2skh/occupancy"
    id_lst = []
    for f in files:
        surface_point_cloud_path = pathrename(f, "_surface_point_cloud.ply")
        occupancy_path = pathrename(f, "_occupancy.npy")
        if os.path.exists(surface_point_cloud_path) and os.path.exists(occupancy_path):
            shutil.copy(surface_point_cloud_path, s_dir)
            shutil.copy(occupancy_path, o_dir)
            id = f.split('/')[-2:]
            id = id[0] + '/' +id[1]
            id = id.split('.')[0]
            id_lst.append(id)
        else:
            continue
    
    j = {"test":id_lst}
    dict2json(j, "/data/wc/extrude_net/data/pc2skh/test.json")
    print(len(id_lst))