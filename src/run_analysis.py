# ----------------------------------------------------------------------------------------
# This code is designed to run the entire analysis pipeline on the Adriot cluster
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

from run_pipeline import run_all_corpora_cnvkt
import sys
import os

sys.path.append(os.path.abspath(".."))

convokit_root = "../../Thesis-Data/Convokit"
run_all_corpora_cnvkt(convokit_root)