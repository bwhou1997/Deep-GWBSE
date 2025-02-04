import numpy as np
import matplotlib.pyplot as plt
import h5py as h5


mol_name = "C2H3"
Aij_index = 0
spin = 0
norb = 4

with h5.File('test_result.h5','r') as f:
    H_pred = f[mol_name+'/H_pred'][()]
    label = f[mol_name+'/label'][()]
    mask = f[mol_name+'/mask'][()]

H_pred = H_pred* mask


vmax = abs(label).max()*0.6

# use imshow to compare label and H_pred

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
im1 = plt.imshow(np.real(label[Aij_index].reshape(norb, norb)), vmax=vmax, vmin=-vmax, cmap='RdBu_r')
# im1 = plt.imshow(np.real(label), vmax=vmax, vmin=-vmax, cmap='RdBu_r')
plt.colorbar(im1, shrink=0.72)
plt.title('H label-'+mol_name)
plt.xticks(range(norb))
plt.yticks(range(norb))

plt.subplot(1,2,2)
im2 = plt.imshow(np.real(H_pred[Aij_index].reshape(norb, norb)), vmax=vmax, vmin=-vmax, cmap='RdBu_r')
# im2 = plt.imshow(np.real(H_pred), vmax=vmax, vmin=-vmax, cmap='RdBu_r')
plt.colorbar(im2, shrink=0.72)
plt.title('H pred (masked)-'+mol_name)
plt.xticks(range(norb))
plt.yticks(range(norb))

plt.savefig('H_pred_'+mol_name+'.png')

# print eigenvalues
H_block_label = label[Aij_index].reshape(norb, norb)
H_block_pred = H_pred[Aij_index].reshape(norb, norb)
print('label', np.sort(np.linalg.eigvals(H_block_label)))
print('pred:', np.sort(np.linalg.eigvals(H_block_pred)))






