import os
import h5py as h5
import numpy as np
import sys
from tqdm import tqdm

#rewrite this line later --Xian
sys.path.append('/pscratch/sd/x/xianxu/Deep-GWBSE/Deep-GWBSE/src')
from interface import wfn
#
def calculate_wfn_r(wf,hovb, nv, nc):
    wf.get_wfn_g_in_grid()
    wfn_r = np.zeros_like(wf.wfn_nk_ggrid[hovb-nv:hovb+nc])
    for ik in tqdm(range(wf.nk)):
        wf_qp_g = wf.wfn_nk_ggrid[hovb-nv:hovb+nc, ik]
        wf_qp_r = np.fft.ifftn(wf_qp_g, s=wf.FFTgrid, norm='forward', axes=(1,2,3)) / np.sqrt(np.prod(wf.FFTgrid))
        wfn_r[:, ik] = wf_qp_r
    wfn_r = wfn_r.transpose(1,0,2,3,4)
    return wfn_r

def hovb_finder(wfn_file):
    with h5.File(wfn_file, 'r') as f:
        ifmax = f['mf_header/kpoints/ifmax'][()]
        nspin = f['mf_header/kpoints/nspin'][()]
        nk = f['mf_header/kpoints/nrk'][()]
        hovb =int(np.ceil(np.sum(ifmax)/nspin/nk))
    return hovb

def band_indices(nk,nv,nc):
    band_ind = np.zeros((nk,nv+nc,1),dtype=int)
    for ik in range(nk):
        for iv in range(nv):
            band_ind[ik,iv,0] = iv - nv
        for ic in range(nc):
            band_ind[ik,ic+nv,0] = ic + 1

    return band_ind

def kpt_generator(wfn_file,nb):
    with h5.File(wfn_file, 'r') as f:
        rk = f['mf_header/kpoints/rk'][()]
    kpt = np.repeat(rk[:,np.newaxis,:],nb,axis=1)
    return kpt

def weight_generator(wf,nb):
    w = wf.k_weights
    nk = wf.nk
    weight = np.repeat(w[:,np.newaxis,np.newaxis],nb,axis=1)
    return weight


def preprocessor(preprocess_folder, processed_folder, processed_file_name, nv, nc):

    key_list = []
    path_list = []
    wfn_r_dic = {}
    band_indices_dic = {}
    kpt_dic = {}
    weight_dic = {}

    # walk through the preprocess_folder
    for subdir in os.listdir(preprocess_folder):
        key_list.append(subdir)
        path_list.append(os.path.join(preprocess_folder, subdir, '02-wfn/wfn.h5'))

    # calculate all datas
    for i in range(len(key_list)):
        wfn_file = path_list[i]
        wf = wfn(wfn_file)
        hovb = hovb_finder(wfn_file)
        # wfn_r
        wfn_r_dic[key_list[i]] = calculate_wfn_r(wf,hovb, nv, nc) # [nk nb ngx ngy ngz]
        # band_indices
        band_indices_dic[key_list[i]] = band_indices(wf.nk, nv, nc)
        #kpt
        kpt_dic[key_list[i]] = kpt_generator(wfn_file, nv+nc)
        #weight
        weight_dic[key_list[i]] = weight_generator(wf, nv+nc)


    # write to h5 file 
    with h5.File(os.path.join(processed_folder, processed_file_name), 'w') as f:
        for i in range(len(key_list)):
            f.require_group(key_list[i])
            f.create_dataset(f'{key_list[i]}/wfn_r', data=wfn_r_dic[key_list[i]])
            f.create_dataset(f'{key_list[i]}/band_indices', data=band_indices_dic[key_list[i]])
            f.create_dataset(f'{key_list[i]}/kpt', data=kpt_dic[key_list[i]])
            f.create_dataset(f'{key_list[i]}/weight', data=weight_dic[key_list[i]])
        f.create_dataset('all_keys', data=np.array(key_list, dtype='S'))


preprocess_folder = '/pscratch/sd/x/xianxu/Deep-GWBSE/Deep-GWBSE/src/DGB_dataloader/DGB_test_examples'
processed_folder = '/pscratch/sd/x/xianxu/Deep-GWBSE/Deep-GWBSE/src/DGB_dataloader/processed_data_dir_test'
processed_file_name = 'processed_data.h5'

nv = 1
nc = 1

preprocessor(preprocess_folder, processed_folder, processed_file_name, nv, nc)