# ----------------------------------------------------------------------------------------
# This code is designed to download and pre-process the Reddit data to be used in analysis
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# general imports
import pandas as pd
from datetime import datetime
from convokit import Corpus, download
from tqdm import tqdm
import nltk
import re

# set up nltk tokenizers
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
nltk.download('punkt_tab')

# set up nltk lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import pos_tag
from nltk.corpus import wordnet

# ----------------------------------------------------------------------------------------
# Format Data
# ----------------------------------------------------------------------------------------
corpus = Corpus(filename=download("subreddit-Cornell"))

def corpus_to_df(corpus):
    '''Function to convert the convokit corpus to a pandas dataframe structure.'''

    data = []
    for utt in corpus.iter_utterances():
        # only consider utterances with timestamps and text
        if hasattr(utt, "timestamp") and utt.text:
            # convert timestamp from seconds since 1/1/1970 to datetime
            t = datetime.fromtimestamp(int(utt.timestamp))

            data.append({
                "utterance_id": utt.id,
                "speaker_id": utt.speaker.id,
                "text": utt.text,
                "timestamp": t
            })

    df = pd.DataFrame(data)
    return df

# ----------------------------------------------------------------------------------------
# Global Variables for DF-Level Cleaning
# ----------------------------------------------------------------------------------------
BOT_TEXT_PATTERNS = [
    r"\bi am a bot\b",
    r"\bthis (comment|post) was (posted|left by) a bot",
    r"\bthis reply was generated automatically",
    r"[\^*]*beep(?:\s+beep)?[\^*]*\s+[\^*]*boop(?:\s+boop)?[\^*]*"
]

BOT_TEXT_RE = re.compile("|".join(BOT_TEXT_PATTERNS), flags=re.IGNORECASE)
URL_RE = re.compile(r"http\S+|www\.\S+")
HAS_LETTER_RE = re.compile(r"[A-Za-z]")


# ----------------------------------------------------------------------------------------
# Pre-Processing helper functions
# ----------------------------------------------------------------------------------------
def clean_text(text):
    '''Helper function to clean text by removing urls and other undesirable features.'''

    # remove urls
    text = URL_RE.sub("", text)

    return text

def tokenize(text):
    '''Helper function to tokenize social media text. Note that the TweetTokenizer 
    preserves mentions, contractions, and other social media-specific structures'''

    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)

    return tokens

def get_wordnet_pos(treebank_tag):
    '''Helper function to map treebank-based POS tags to wordnet POS tags.'''
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    
    # otherwise default to noun
    else:
        return wordnet.NOUN
    
def lemmatize(tokens):
    '''Helper function to lemmatize tokens.'''

    tagged = pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(tok, get_wordnet_pos(tag)) for tok, tag in tagged]

    return lemmatized_tokens

def clean_tokens_lexical(text):
    '''Helper function that tokenizes text, cleans tokens by removing punctuation and numbers
    for purely lexical analysis, and returns the cleaned, lemmatized tokens.'''
    
    # clean text
    text = clean_text(text)

    # tokenize text
    tokens = tokenize(text)

    # clean tokens
    cleaned = []
    for tok in tokens:
        # skip over punctuation
        if re.match(r'^\W+$', tok):
            continue

        # only keep alphabetic tokens, including contractions
        if re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", tok):
            cleaned.append(tok.lower())

    # lemmatize clean tokens
    lemmatized = lemmatize(cleaned)

    return lemmatized

# ----------------------------------------------------------------------------------------
# Pre-Processing Function
# ----------------------------------------------------------------------------------------
def preprocess_df(df):
    '''Function that pre-processes the data in a given dataframe by removing
    deleted/removed utterances, bot utterances, and utterances not containing letters.
    Then, the remaining textual data is tokenized, lemmatized, and cleaned.'''

    # remove deleted/removed utterances
    df = df[~df["text"].str.lower().isin({"[deleted]", "[removed]"})]

    # remove bot authored utterances
    df = df[~df["text"].str.contains(BOT_TEXT_RE)]

    # remove utterances without a letter
    df = df[df["text"].str.contains(HAS_LETTER_RE)]

    # tokenize
    df["tokens"] = df["text"].apply(tokenize)

    # lemmatize
    df["lemmas"] = df["tokens"].apply(lemmatize)

    # final
    df["final"] = df["text"].apply(clean_tokens_lexical)

    return df