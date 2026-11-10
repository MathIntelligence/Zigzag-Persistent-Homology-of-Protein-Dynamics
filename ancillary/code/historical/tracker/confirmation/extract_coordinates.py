import h5py,numpy as np
from pathlib import Path
H=Path.home()/'mnt'; out=Path('ca'); out.mkdir(exist_ok=True)
for dom in open('domains.txt').read().split():
    if (out/f'{dom}.npz').exists(): continue
    p=H/'zigzagv3VS'/'mdcath'/'data'/f'mdcath_dataset_{dom}.h5'
    if not p.exists(): p=H/'Downloads'/f'mdcath_dataset_{dom}.h5'
    try:
        g=h5py.File(p,'r')[dom]
        pdb=g['pdbProteinAtoms'][()]; pdb=pdb.decode() if isinstance(pdb,bytes) else str(pdb)
        lines=[l for l in pdb.splitlines() if l.startswith(('ATOM  ','HETATM'))]
        ca=[i for i,l in enumerate(lines) if l[12:16].strip()=='CA']
        d={'ref':np.asarray(g['320']['0']['coords'][0][ca],np.float32)}
        for T in ('413','450'):
            for rep in sorted(g[T],key=int): d[f'{T}_{rep}']=np.asarray(g[T][rep]['coords'][:,ca,:],np.float32)
        np.savez(out/f'{dom}.npz',**d); print(dom,len(ca),flush=True)
    except Exception as e: print('FAIL',dom,e,flush=True)
print('ALLDONE',flush=True)
