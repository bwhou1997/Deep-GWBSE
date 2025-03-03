from torch.utils.data import DataLoader, Dataset
import h5py as h5

# Example dataset
class DGB_Dataset(Dataset):
    def __init__(self, processed_data_h5):
        f = h5.File(processed_data_h5, 'r')
        self.h5file = f
        self.all_keys = f['all_keys'][()]

    def __len__(self):
        return len(self.all_keys)

    def __getitem__(self, idx):
        return self.all_keys[idx], self.h5file[self.all_keys[idx]]['wfn_r'][()], self.h5file[self.all_keys[idx]]['band_indices'][()], self.h5file[self.all_keys[idx]]['kpt'][()], self.h5file[self.all_keys[idx]]['weight'][()]
    

processed_data_h5_test = '/pscratch/sd/x/xianxu/Deep-GWBSE/Deep-GWBSE/src/DGB_dataloader/processed_data_dir_test/processed_data.h5'
dataset_test = DGB_Dataset(processed_data_h5_test)


dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)

# Iterate through the DataLoader
for keyname, wfn_r, band_indices, kpt, weight in dataloader:
    print(keyname)
    print(wfn_r.shape)
    print(band_indices.shape)
    print(kpt.shape)
    print(weight.shape)


