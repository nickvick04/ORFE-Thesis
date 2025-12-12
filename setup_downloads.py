# ----------------------------------------------------------------------------------------
# This code is designed to download the required models for libraries like spacy
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

# necessary package imports
import subprocess
import sys

def install_models():
    '''Function to download all required models for data analysis.'''
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

if __name__ == "__main__":
    install_models()