import os
import glob
import json
from tqdm import tqdm
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
        files = [path.strip() for path in lines]
    else:
        files = []
        print("Gathering all obj files...")
        files = glob.glob("%s/**/*.obj" % DATASET_PATH, recursive=True)
        with open(CACHE_PATH, "w") as f:
            for path in files:
                f.write(path + "\n")
    return files

def cp_file(f, suf1, suf2):
    file1_path = pathrename(f, suf1)
    file2_path = pathrename(f, suf2)
    if os.path.exists(file1_path) and os.path.exists(file2_path):
        id = f.split('/')[-2:]
        id = id[0] + '/' +id[1]
        id = id.split('.')[0]
        IDLST.append(id)
        F_LST.append((file1_path ,file2_path))
    return

if __name__ == "__main__":
    CACHE_PATH = "/data/wc/Points2sketch/DATABASE/pc2skh_3d/data_filter.txt"
    DATASET_PATH = ""
    data_base_dir = "/data/wc/extrude_net/data/pc2skh/extrudenet"
    SUFF1 = "_surface_point_cloud.ply"
    # SUFF1 = "_sdf.vox"
    SUFF2 = "_occupancy.npy"

    if not os.path.exists(data_base_dir):
        os.makedirs(data_base_dir)
    files = get_all_obj_path(use_cache=True)
    F_LST = []
    IDLST = []
    pbar = tqdm(files)
    for f in pbar:
        cp_file(f,SUFF1,SUFF2,)


    JSON_FILE = os.path.join(data_base_dir, "test.json") 
    j = {"test":IDLST}
    dict2json(j, JSON_FILE)
    print(len(IDLST))
    
    num = len(F_LST)

    text_file_path_train = os.path.join(data_base_dir, "train.txt")
    text_file_path_val   = os.path.join(data_base_dir, "val.txt")
    text_file_path_test  = os.path.join(data_base_dir, "test.txt")

    train_lst = F_LST[:int(num*0.8)]
    with open(text_file_path_train, 'w') as f:
        for f1, f2 in train_lst:
            f.write(f1 + ' ' + f2 + '\n')
    
    val_lst   = F_LST[int(num*0.8):int(num*0.9)]
    with open(text_file_path_val, 'w') as f:
        for f1, f2 in val_lst:
            f.write(f1 + ' ' + f2 + '\n')
            
    test_lst  = F_LST[int(num*0.9):]
    with open(text_file_path_test, 'w') as f:
        for f1, f2 in test_lst:
            f.write(f1 + ' ' + f2 + '\n')

