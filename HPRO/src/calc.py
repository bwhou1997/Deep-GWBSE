#!/usr/bin/env python

from HPRO import PW2AOkernel
import json

with open('calc.json','r') as file:
    kwargs = json.load(file)

kernel = PW2AOkernel(
    lcao_interface=kwargs['lcao_interface'],
    lcaodata_root=kwargs['lcaodata_root'],
    hrdata_interface=kwargs["hrdata_interface"],
    vscdir=kwargs["vscdir"],
    upfdir=kwargs["upfdir"],
    ecutwfn=kwargs["ecutwfn"],
)

kernel.run_pw2ao_rs(kwargs['outdir'])
