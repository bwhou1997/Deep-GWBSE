#%%
from data import ManyBodyData

ManyBodyData(flows_dir='../../examples/nc300/semiconductor/flows-nc300/', dataset_dir='./dataset', dataset_type='WFN', dataset_fname='dataset_WFN.h5',
                          load_dataset=False, cell_slab_truncation=30, useWignerXY=True, AngstromPerPixel=0.1,
                          AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
                          nc_wfn=40, nv_wfn=2)