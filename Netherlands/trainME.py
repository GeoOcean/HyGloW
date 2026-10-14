import os

from utils import metamodel as mm

os.environ["OMP_NUM_THREADS"] = "8"  # avoid MKL using all cores

mm.fit_all(gp_epochs=5000)
