# ----------------------------------------------------------------------------------------
# This code is designed to compute measures of lexical complexity from Reddit data
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# general imports 
import numpy as np
import pandas as pd
from convokit import Corpus, download
import matplotlib.pyplot as plt

# imports specific to lexical measures
import re
from wordfreq import zipf_frequency
from lexical_diversity import lex_div as ld
from collections import Counter

# import data processing functions
from src.data_preprocessing import corpus_to_df, preprocess_df

# ----------------------------------------------------------------------------------------
# Lexical Analysis Functions
# ----------------------------------------------------------------------------------------
def mtld_score(clean_tokens):
    '''Function that returns the MTLD score for a given set of cleaned, lemmatized tokens.
    A higher MTLD score indicates higher lexical diversity.'''

    # compute mtld
    mtld_score = ld.mtld(clean_tokens)

    return mtld_score

def yules_K(clean_tokens):
    '''Function that returns Yule's characteristic constant K for a given set of 
    cleaned, lemmatized tokens. A lower Yule's K indicates higher lexical diversity.'''

    N = len (clean_tokens)
    if N == 0:
        return 0.0
    
    # count how many times each word occurs
    freq_counts = Counter(clean_tokens)

    # create V_i
    V = Counter(freq_counts.values()) # keys are frequencies, values are counts of types

    # compute sum of i^2 * V_i
    sum_isq_vi = sum((i**2) * Vi for i, Vi in V.items())

    # apply Yule's formula
    K = 10000 * ((sum_isq_vi - N)/(N**2))

    return K

def zipf_score(clean_tokens):
    '''Returns the average frequency score (higher -> more frequent) based on the Zipf scale
    for a given set of cleaned, lemmatized tokens. A higher Zipf score corresponds to higher
    lexical diversity.'''
    
    # compute Zipf parameter Z for each word
    zipf_values = [zipf_frequency(word, 'en') for word in clean_tokens]

     # if there are no words, return a default value
    if len(zipf_values) == 0:
        return np.nan

    # find the average zipf parameter
    zipf_score = np.mean(zipf_values)

    return zipf_score

# build aoa_dict: word -> average age of acquisition
aoa_df = pd.read_csv("../Data/KupermanAoAData.csv")
aoa_dict = dict(zip(aoa_df["word"], aoa_df["rating_mean"]))

def aoa_score(clean_tokens, aoa_dict=aoa_dict):
    '''Returns the average age of acquisition score for a set of cleaned, lemmatized tokens.
    A higher mean AoA score indicates higher lexical difficulty.'''
    
    # extract aoa value only if the word is in the AoA dict
    aoa_values = [aoa_dict[word] for word in clean_tokens if word in aoa_dict]

    # if there are no words, return a default value
    if len(aoa_values) == 0:
        return np.nan
    
    # average the aoa values across all words
    aoa_score = np.mean(aoa_values)

    return aoa_score

# import NAWL list of academic words
nawl_list = pd.read_csv("../Data/nawl_cleaned.csv")

def nawl_ratio(clean_tokens, nawl_list=nawl_list):
    '''Returns the ratio of words present in the NAWL given a set of cleaned, lemmatized tokens.
    A higher NAWL ratio indicates higher academic lexical sophistication.'''

    # compute the total number of words
    total_num_words = len(clean_tokens)

    # if there are no tokens, return null value
    if total_num_words == 0:
        return np.nan

    num_nawl_words = 0
    # compute the number of values in NAWL
    for word in clean_tokens:
        if word in nawl_list:
            num_nawl_words += 1

    # compute the nawl ratio
    nawl_ratio = num_nawl_words / total_num_words

    return nawl_ratio

# compute the relevant metrics for each utterance
def compute_lexical_vals(df):
    '''Function that applies all of the relevant lexical analysis functions to the cleaned tokens.'''

    print("Computing MTLD values...")
    df["mtld_score"] = df["final"].apply(mtld_score)

    print("Computing Yules K values...")
    df["yules_k"] = df["final"].apply(yules_K)

    print("Computing Zipf score values values...")
    df["zipf_score"] = df["final"].apply(zipf_score)

    print("Computing AOA score values...")
    df["aoa_score"] = df["final"].apply(aoa_score)

    print("Computing NAWL ratio values...")
    df["nawl_ratio"] = df["final"].apply(nawl_ratio)