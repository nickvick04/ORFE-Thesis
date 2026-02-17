# ----------------------------------------------------------------------------------------
# This code is designed to download the required models for libraries like spacy
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

# necessary package imports
import subprocess
import sys
import nltk
import stanza

def install_models():
    '''Download all required NLP models BEFORE running SLURM jobs.'''

    print("Downloading spaCy model...")
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
    )

    print("Downloading NLTK resources...")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")
    nltk.download("treebank")

    print("Downloading Stanza English model...")
    stanza.download("en")

    print("All models downloaded successfully.")

if __name__ == "__main__":
    install_models()