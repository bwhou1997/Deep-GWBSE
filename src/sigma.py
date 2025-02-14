import numpy as np
import numpy as np
import h5py as h5

from scipy.io import FortranFile


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


class v_file:
    """
    modified from /HPRO/bgwio.py
    v: VXC, VSC files
    """
    def __init__(self, vscfile):
        self.fname = vscfile
        self.file = FortranFile(vscfile, 'r')
        self._read = False
        self._set_vcsg = False
        self.read_v()
        self.file.close()

    def read_v(self):        
        assert not self._read
        rec = self.file.read_record('S1').tobytes().decode()
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
    
    def write_v(self):
        outfile = self.fname + '.new'
        assert self._read
        with FortranFile(outfile, 'w') as f:
            # Write header information
            rec = (self.stitle.ljust(32) + self.sdate.ljust(32) + self.stime.ljust(32)).encode()
            f.write_record(np.array(list(rec), dtype='S1'))
            
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
            print(self.vscg)
            f.write_record((self.vscg*2).astype('c16'))

    @property
    def vscg(self):
        return self._vscg

    @vscg.setter
    def vscg(self, value):
        assert self._read # Only allow setting if the file has been read
        self.old_vcsg = self._vscg
        print('reset vscg')
        self._vscg = value
        if np.allclose(self.old_vcsg, self._vscg):
            print('vscg is the same')
        else:
            print('vscg is different')
        self._set_vcsg = True


class wfn_file:
    def __init__(self, wfn_file_h5):
        pass


# class eqp(eqp_file):
#     def __init__(self, eqp_file):
#         super().__init__(eqp_file)

#     def eqp2vxc(self,):
#         pass


if __name__ == '__main__':
    eqp_dat = './test/eqp.dat'
    eqp = eqp_file(eqp_dat)

    # wfn = wfn_file('./test/wfn.h5')

    vxc = v_file('./test/VXC')
    vxc.vscg = vxc.vscg + 10
    vxc.write_v()

    vxc_new = v_file('./test/VXC.new')

    print('\n')
    for attr in dir(vxc_new):
        if not attr.startswith('_'):
            print(attr, getattr(vxc, attr))
            print(attr, getattr(vxc_new, attr))
            print('\n')
