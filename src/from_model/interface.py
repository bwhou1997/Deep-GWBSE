import numpy as np
import numpy as np
import h5py as h5
import os
import matplotlib.pyplot as plt
from scipy.io import FortranFile
from model_util import H5ls, time_watch, memory_watch, eV2Ry
from tqdm import tqdm

class eqp_file():
    """
    These object decompose eqp.dat into data_DFT, data_GW, klist and spin_list
    """
    def __init__(self,fname):
        print('Reading eqp')
        self.fname = fname
        self.read_eqp()
        # These variables are initialized
        # (1) nbnd, nk
        # (2) data_GW, data_DFT -> (nk, nbnd)
        # (3) band_index, spin ->(nband,);
        # (4) klist -> (nk,)

        # self.write()

    def read_eqp(self):
        f = open(self.fname, 'r')
        lines = f.readlines()
        f.close()

        self.nbnd = int(lines[0].split()[-1])
        self.nk = int(len(lines) / (self.nbnd + 1))

        print('number of bands:', self.nbnd)
        print('number of kpoints:', self.nk)

        self.data_GW = np.zeros((self.nk, self.nbnd))
        self.data_DFT = np.zeros((self.nk, self.nbnd))
        self.band_index = np.zeros((self.nk, self.nbnd),dtype=int)
        self.spin_index = np.zeros((self.nk, self.nbnd),dtype=int)
        self.klist = []

        # get
        line = 0
        for i in range(self.nk):
            temp_k = lines[line]
            self.klist.append("  ".join(temp_k.split()[:3]))
            # print(" ".join(temp_k.split()[:3]))
            line += 1
            for j in range(self.nbnd):
                self.band_index[i,j] = int(lines[line].split()[1])
                self.spin_index[i,j] = int(lines[line].split()[0])
                self.data_DFT[i,j] = lines[line].split()[-2]
                self.data_GW[i,j] = lines[line].split()[-1]
                line += 1
        self.data_DFT = np.around(self.data_DFT,9)


    def write_eqp(self):
        f_new = open('eqp_new.dat','w')
        line = 0
        for i in range(self.nk):
            f_new.write('  '+self.klist[i]+'      %s'%self.nbnd+'\n')
            line += 1
            for j in range(self.nbnd):
                f_new.write('       %s     %s    %.9f    %.9f\n' % (self.spin_index[i,j], self.band_index[i,j], self.data_DFT[i,j], self.data_GW[i,j]))
                # print('%s %s %s %s' % (self.spin_index[j], self.band_index[j], self.data_DFT[i,j], self.data_GW[i,j]))
                line += 1

    def plot_eig(self):
        f = open('band.dat', 'w')
        for i in range(self.nbnd):
            for j in range(self.nk):
                f.write("%s %s %s\n" % (j + 1, self.data_DFT[j, i], self.data_GW[j, i]))
            f.write('\n')
        f.close()

class vloc:
    """
    modified from /HPRO/bgwio.py
    vscfile: VXC, VSC files
    """
    def __init__(self, vscfile):
        self.fname = vscfile
        self.file = FortranFile(vscfile, 'r')
        self._read = False
        self._set_vcsg = False
        self.read_v()


    def read_v(self):        
        assert not self._read
        rec = self.file.read_record('S1')
        self.stitle_sdate_stime = rec.copy()
        rec = rec.tobytes().decode()
        self.stitle = rec[0:32].rstrip(' ')
        self.sdate = rec[32:64].rstrip(' ')
        self.stime = rec[64:96].rstrip(' ')
        # print(self.stitle, self.sdate, self.stime)
        
        rec = self.file.read_record('i4', 'i4', 'i4', 'i4', 'i4', 'f8')
        rec = map(lambda x: x.item(), rec)
        self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat, self.ecutrho = rec # nk = nk_g / ns
        # print(self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat, self.ecutrho)
        
        rec = self.file.read_record(*(['i4']*3))
        rec = map(lambda x: x.item(), rec)
        self.nr1, self.nr2, self.nr3 = rec

        rec = self.file.read_record('f8')
        self.omega = rec[0] # cell volume, in bohr^3
        self.alat = rec[1] # in bohr
        self.at = rec[2:11].reshape(3, 3)
        self.adot = rec[11:20].reshape(3, 3)
        # print(omega, alat)
        # print(at)
        # print(adot) # ai dot aj, in bohr^2
        
        rec = self.file.read_record('f8')
        self.recvol = rec[0]
        self.tpiba = rec[1] # in bohr-1
        self.bg = rec[2:11].reshape(3, 3) # reciprocal lattice vecs
        self.bdot = rec[11:20].reshape(3, 3)
        # print(recvol, tpiba)
        # print(bg)
        # print(adot)
        
        rec = self.file.read_record('i4')
        self.rotmat = rec.reshape(self.ntran, 3, 3) # rotation matrices
        # print(rotmat)
        
        rec = self.file.read_record('f8')
        self.frac_tran = rec.reshape(self.ntran, 3) # fractional translations
        # print(frac_tran)
        
        rec = self.file.read_record(*(['f8']*3 + ['i4'])*self.nat)
        self.tau = np.empty((self.nat, 3), dtype=float) # (nat, 3) atomic positions (alat)
        self.atomic_number = np.empty(self.nat, dtype=int) # (nat) atomic numbers
        for iat in range(self.nat):
            for d in range(3):
                self.tau[iat, d] = rec[iat*4+d]
            self.atomic_number[iat] = rec[iat*4+3]
        # print(tau) # alat
        # print(atomic_number)

        
        nrecord = self.file.read_record('i4').item()
        self.ng_g = self.file.read_record('i4').item() # number of charge_density g-vecs
        self.g_g = self.file.read_record('i4').reshape(self.ng_g, 3)
        # print(nrecord, ng_g)

        nrecord = self.file.read_record('i4').item()
        self.ng_g = self.file.read_record('i4').item() # number of charge_density g-vecs
        self._vscg = self.file.read_record('c16').reshape(self.nsf, self.ng_g) * 0.5 # Ry->Ha

        self._read = True
        FFTgrid = np.array([self.nr1, self.nr2, self.nr3])
        _, self.g_g_full = np.divmod(self.g_g, FFTgrid)
        self.FFTgrid = FFTgrid
        self.file.close()

        
    def write_v(self, outfile=None):
        assert self._read
        print('Writing VXC/VSC')
        self.check_reset()
        if outfile is None:
            outfile = self.fname + '.new'
        with FortranFile(outfile, 'w') as f:
            # Write header information
            # rec = (self.stitle.ljust(32) + self.sdate.ljust(32) + self.stime.ljust(32)).encode()
            # rec = (self.stitle.ljust(32)+ self.sdate.ljust(32) + self.stime.ljust(32)).encode()
            # f.write_record(np.array(list(rec), dtype='S1'))
            f.write_record(self.stitle_sdate_stime)
            
            # Write integer and float metadata
            f.write_record(np.array([self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat], dtype='i4'),
                        np.array([self.ecutrho], dtype='f8'))
            f.write_record(np.array([self.nr1, self.nr2, self.nr3], dtype='i4'))
            
            # Write lattice and atomic data
            f.write_record(np.array([self.omega, self.alat] + self.at.flatten().tolist() + self.adot.flatten().tolist(), dtype='f8'))
            f.write_record(np.array([self.recvol, self.tpiba] + self.bg.flatten().tolist() + self.bdot.flatten().tolist(), dtype='f8'))
            f.write_record(self.rotmat.flatten().astype('i4'))
            f.write_record(self.frac_tran.flatten().astype('f8'))
            
            # Write atomic positions and numbers
            # TODO
            flat_data = []
            for iat in range(self.nat):
                flat_data.extend(np.array(self.tau[iat], dtype='f8'))  # Extend with 3 components of tau
                flat_data.append(np.array(self.atomic_number[iat], dtype='i4'))  # Append atomic number
            f.write_record(*flat_data)
            
            # Write g-vector information
            f.write_record(np.array([1], dtype='i4'))  # Dummy value for compatibility
            f.write_record(np.array([self.ng_g], dtype='i4'))
            f.write_record(self.g_g.astype('i4'))
            
            f.write_record(np.array([1], dtype='i4'))  # Another dummy value
            f.write_record(np.array([self.ng_g], dtype='i4'))
            # print(self.vscg) 
            f.write_record((self._vscg*2).astype('c16')) # Ha->Ry

    def get_vlocr(self, plotXY=False):
        assert self._read
        self.check_reset()
        vlocg_full = np.zeros(self.FFTgrid, dtype='c16')
        vlocg_full[self.g_g_full[:, 0], self.g_g_full[:, 1], self.g_g_full[:, 2]] = self.vscg
        vlocr = np.fft.ifftn(vlocg_full, s=self.FFTgrid, norm='forward')
        assert np.max(np.abs(vlocr.imag)) < 1e-6
        vlocr = vlocr.real
        if plotXY:
            plt.figure()
            plt.imshow(vlocr.sum(axis=2))
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Vlocr Reset: {self._set_vcsg}')
            plt.show()
        # print('vlocr shape:', vlocr.shape)
        self.vlocg_full = vlocg_full
        self.vlocr = vlocr
        return vlocr

    def check_reset(self):
        if self._set_vcsg:
            if np.allclose(self.old_vcsg, self._vscg):
                print('vscg is reset: vscg is the same')
            else:
                print('vscg is reset: vscg is different')
        else:
            print('vscg is not reset')

    def IO_test(self):
        assert self._read
        self.write_v('./VXC.test')
        v_test = vloc('./VXC.test')
        for attr in dir(v_test):
            # print(attr, getattr(self, attr), getattr(v_test, attr))
            if not attr.startswith('_'):
                if type(getattr(self, attr)) == np.ndarray:
                    # dtype != s1
                    if getattr(self, attr).dtype != 'S1':
                        assert np.allclose(getattr(self, attr), getattr(v_test, attr))
                    # assert np.allclose(getattr(self, attr), getattr(v_test, attr))
                    pass
        os.remove('./VXC.test')
        print('IO test VXC/VSC: pass')

    @property
    def vscg(self):
        return self._vscg

    @vscg.setter
    def vscg(self, value):
        assert self._read # Only allow setting if the file has been read
        assert value.shape == self._vscg.shape 
        self.old_vcsg = self._vscg
        self._vscg = value
        if np.allclose(self.old_vcsg, self._vscg):
            print('resetting: vscg is the same')
        else:
            print('resetting: vscg is different')
        self._set_vcsg = True

class wfn:
    def __init__(self, wfn_file_h5):
        # Status flags
        self._read_header = False
        self._get_wfn_g = False
        self._get_wfn_r = False

        # Open the file
        self.wfn_file_h5 = wfn_file_h5
        self.wfn_file = h5.File(wfn_file_h5, 'r')

        # Get the names of the datasets
        h5ls = H5ls()
        self.wfn_file.visititems(h5ls)   
        self.names = h5ls.names

        # Get the header information
        self.crystal = {}
        self.gspace = {}
        self.kpoints = {}
        self.symmetry = {}
        self.wfns = {}
        self.read_header()

        # Close the file
        self.wfn_file.close()

    def read_header(self):
        for name in self.names:
            if 'crystal' in name.split('/'):
                self.crystal[name.split('/')[-1]] = self.wfn_file[name][()]
            elif 'gspace' in name.split('/'):
                self.gspace[name.split('/')[-1]] = self.wfn_file[name][()]
            elif 'kpoints' in name.split('/'):
                self.kpoints[name.split('/')[-1]] = self.wfn_file[name][()]
            elif 'symmetry' in name.split('/'):
                self.symmetry[name.split('/')[-1]] = self.wfn_file[name][()]
            elif 'wfns/gvecs' == name:
                self.wfns[name.split('/')[-1]] = self.wfn_file[name][()]
            else:
                pass
        cum_sum = np.cumsum(np.concatenate((np.array([0]),self.kpoints['ngk'])))
        self.nkg_slice = [[cum_sum[i], cum_sum[i+1]] for i in range(self.kpoints['nrk'])]
        self._read_header = True

        self.nk = self.kpoints['occ'].shape[1]
        self.nb = self.kpoints['occ'].shape[2]
        self.g_g = self.wfns['gvecs']
        self.el = self.kpoints['el'][0]
        self.k_weights = self.kpoints['w']
        self.FFTgrid = self.gspace['FFTgrid']

    
    @time_watch
    @memory_watch()
    def get_wfn_g_in_grid(self):
        """
        Calculate the wavefunction in full G-grid   
        Input: 
        Output:
            create self.wfn_nk_ggrid: (nb, nk, FFTgrid[0], FFTgrid[1], FFTgrid[2])
        """
        assert self._read_header
        print('Loading wfn.h5...')
        # Be careful with the shape of the wavefunction coefficients, 
        f = h5.File(self.wfn_file_h5, 'r')
        self.wfns['coeffs'] = f['wfns/coeffs'][()]
        f.close()

        FFTgrid = self.gspace['FFTgrid']
        self.wfn_nk_ggrid = np.zeros((self.nb, self.nk, FFTgrid[0], FFTgrid[1], FFTgrid[2]), dtype='c16')
        _, g_g_full = np.divmod(self.g_g, FFTgrid)

        # print('Raw Wavefunction:', self.wfns['coeffs'].nbytes/1024/1024, 'MB')
        for ib in tqdm(range(self.nb), desc='Building Wavefunction in Full G-grid'):
            wfn_b = self.wfns['coeffs'][ib,0,:,0] + self.wfns['coeffs'][ib,0,:,1]*1j
            for ik in range(self.nk):
                gx = g_g_full[self.nkg_slice[ik][0]:self.nkg_slice[ik][1], 0]
                gy = g_g_full[self.nkg_slice[ik][0]:self.nkg_slice[ik][1], 1]
                gz = g_g_full[self.nkg_slice[ik][0]:self.nkg_slice[ik][1], 2]
                self.wfn_nk_ggrid[ib,ik,gx,gy,gz] = wfn_b[self.nkg_slice[ik][0]:self.nkg_slice[ik][1]]
        # print('Full G-grid wavefunction:', self.wfn_nk_ggrid.nbytes/1024/1024, 'MB')
        # Norm Check
        self._get_wfn_g = True

    def get_wfn_r_in_grid(self):
        pass



def eqp2vsc_hat(eqp_dat='./test_data/eqp_full.dat', vsc_file='./test_data/VSC', wfn_file='./test_data/wfn_full.h5'): 
    nqp = 8 # number of quasiparticle energies considered

    eqp = eqp_file(eqp_dat)
    vsc = vloc(vsc_file)
    wf = wfn(wfn_file)
    wf.get_wfn_g_in_grid()

    # corr_nk = np.zeros((wf.nb, wf.nk))
    vxc_corr_full_r = np.zeros((wf.FFTgrid))
    for ik in tqdm(range(wf.nk), desc='Building Sigma-Vxc in real space'):
        # eqp.band_index start with 1
        qp_index = eqp.band_index[ik,:nqp]-1
        qp = eqp.data_GW[ik,:nqp] - eqp.data_DFT[ik,:nqp] # Sigma -Vxc
        # corr_nk[qp_index, ik] = eqp.data_GW[ik] - eqp.data_DFT[ik] # Sigma -Vxc
        wf_qp_g = wf.wfn_nk_ggrid[qp_index, ik] # (nb, ng)
        wf_qp_r = np.fft.ifftn(wf_qp_g, s=wf.FFTgrid, norm='forward', axes=(1,2,3)) / np.sqrt(np.prod(wf.FFTgrid))
        vxc_corr_full_r += np.einsum('b, bxyz->xyz', qp, abs(wf_qp_r)**2) * wf.k_weights[ik] * eV2Ry  # eV -> Ry

    # vxc.get_vlocr(plotXY=True)
    vxc_corr_full_g = np.fft.fftn(vxc_corr_full_r, s=wf.FFTgrid, norm='backward') / np.sqrt(np.prod(wf.FFTgrid))
    vxc_corr_g = vxc_corr_full_g[vsc.g_g_full[:, 0], vsc.g_g_full[:, 1], vsc.g_g_full[:, 2]]
    vsc.vscg = vxc_corr_g + vsc.vscg
    vsc.write_v()
    # print('vlocr shape:', vlocr.shape)

if __name__ == '__main__':
    # eqp2vsc_hat(vsc_file='./test_data/VXC')
    # nqp = 8
    # eqp = eqp_file('./test_data/eqp_full.dat')
    vsc = vloc('../flows/mat-3/02-wfn/VSC')
    wf = wfn('../flows/mat-3/02-wfn/wfn.h5')
    # wf.get_wfn_g_in_grid()




    # vxc_corr_full_r = np.zeros(np.prod(wf.FFTgrid))
    # for ik in tqdm(range(wf.nk), desc='Building Sigma-Vxc in real space'):
    #     # eqp.band_index start with 1
    #     qp_index = eqp.band_index[ik,:nqp]-1
    #     qp = eqp.data_GW[ik,:nqp] - eqp.data_DFT[ik,:nqp] # Sigma -Vxc
    #     # corr_nk[qp_index, ik] = eqp.data_GW[ik] - eqp.data_DFT[ik] # Sigma -Vxc
    #     wf_qp_g = wf.wfn_nk_ggrid[qp_index, ik] # (nb, ng)
    #     wf_qp_nr = np.fft.ifftn(wf_qp_g, s=wf.FFTgrid, norm='forward', axes=(1,2,3)) / np.sqrt(np.prod(wf.FFTgrid))

    #     # <r|Sigma'|r> = wf_qp_rn * <n|Sigma|n>, where Sigma' is an approximation of Sigma
    #     wf_qp_nr_ = ((abs(wf_qp_nr)**2).reshape(nqp,-1))
    #     # wf_qp_rn_pinv = np.linalg.pinv(wf_qp_nr_)
    #     # vxc_corr_full_r += np.einsum('rn, n->r',  wf_qp_rn_pinv, qp) * wf.k_weights[ik] * eV2Ry 
    #     ridge.fit(wf_qp_nr_, qp)
    #     vxc_corr_full_r += ridge.coef_ * wf.k_weights[ik] * eV2Ry


    # vxc_corr_full_r = vxc_corr_full_r.reshape(wf.FFTgrid) # eV -> Ry
    # vxc_corr_full_g = np.fft.fftn(vxc_corr_full_r, s=wf.FFTgrid, norm='backward') / np.sqrt(np.prod(wf.FFTgrid))
    # vxc_corr_g = vxc_corr_full_g[vsc.g_g_full[:, 0], vsc.g_g_full[:, 1], vsc.g_g_full[:, 2]]
    # vsc.vscg = vxc_corr_g + vsc.vscg
    # vsc.write_v()