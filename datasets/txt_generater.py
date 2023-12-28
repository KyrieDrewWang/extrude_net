import os

sur_dir = "/data/wc/extrude_net/data/pc2skh/secad/vox"
occ_dir = "/data/wc/extrude_net/data/pc2skh/secad/occupancy"

DATA_base = "/data/wc/extrude_net/data/pc2skh/secad"

src_suffix = "_sdf.vox"
gt_suffix = "_occupancy.npy"

text_file_path_train = os.path.join(DATA_base, "data_train.txt")  
text_file_path_val  =  os.path.join(DATA_base, "data_val.txt")
text_file_path_test =  os.path.join(DATA_base, "data_test.txt")


s_lst = os.listdir(sur_dir)

s_path_lst = []
o_path_lst = []
for s in s_lst:
    s_path = os.path.join(sur_dir, s)
    o = s.replace(src_suffix, gt_suffix)
    o_path = os.path.join(occ_dir, o)
    if not os.path.exists(o_path):
        continue
    s_path_lst.append(s_path)
    o_path_lst.append(o_path)
    
num = len(s_path_lst)

_s_path_lst = s_path_lst[:int(num*0.8)]
_o_path_lst = o_path_lst[:int(num*0.8)]

textfile = open(text_file_path_train, "w")
for im,mas in zip(_s_path_lst, _o_path_lst):
    textfile.write(im+' '+mas + "\n")
textfile.close()

_s_path_lst = s_path_lst[int(num*0.8):int(num*0.9)]
_o_path_lst = o_path_lst[int(num*0.8):int(num*0.9)]

textfile = open(text_file_path_val, "w")
for im,mas in zip(_s_path_lst, _o_path_lst):
    textfile.write(im+' '+mas + "\n")
textfile.close()

_s_path_lst = s_path_lst[int(num*0.9):]
_o_path_lst = o_path_lst[int(num*0.9):]

textfile = open(text_file_path_test, "w")
for im,mas in zip(_s_path_lst, _o_path_lst):
    textfile.write(im+' '+mas + "\n")
textfile.close()