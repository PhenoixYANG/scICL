import os
def generate_log(ldir):
    if not os.path.exists(ldir):
        os.mkdir(ldir)
    l=len(os.listdir(ldir))+1
    ldir=os.path.join(ldir,f'version_{l}')
    os.mkdir(ldir)
    return ldir

