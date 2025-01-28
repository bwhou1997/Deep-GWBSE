import json as js

class fp_config:
    """
    1. read configure file in fp-input
    2. analyze mat-xx in fp-input
    """
    def __init__(self):
        self.fpconfig_read = False
        self.dir_read = False
    
    def read_fpconfig(self, root='./fp-input'):
        self.root = root
        self.config_fname = root + 'fpconfig.json'
        with open(self.config_fname, 'r') as file:
            self.config = js.load(file)
        print('\n===> fp config <===')
        print(js.dumps(self.config, indent=1, ensure_ascii=False))
        print('===> fp config <===\n')

        # start read
        self.ibnd_min = self.config['ibnd_min']
        self.ibnd_max = self.config['ibnd_max']
        self.ecuteps = self.config['ecuteps']
        self.ngkpt = self.config['ngkpt']
        self.qshift = self.config['qshift']
        self.nbnd = self.config['nbnd'] # nscf
        self.ecutwfc = self.config['ecutwfc']
        self.prefix = self.config['SiH']
        
        # finish reading
        self.read_fpconfig = True
    
    def read_dir(self):
        # analyze fp-input
        # 1. how many mats
        # 2. the order of element {'prefix/stru.cif':[ele1.upf, ele2.upf...]}
        assert self.read_fpconfig
        self.nmat = 0
        self.mats = {} # sort?
        self.mats_pp_order = {}
        self.mats_nelements = {}
        self.mats_nelectron = {}


        self.read_dir = True

    def generate_fpconfig_default(self):
        # todo: for test reason
        pass
