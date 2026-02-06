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
                "raw_text": utt.text,
                "timestamp": t
            })

    df = pd.DataFrame(data)
    return df

# ----------------------------------------------------------------------------------------
# Global Variables for DF-Level Cleaning
# ----------------------------------------------------------------------------------------
BOT_TEXT_PATTERNS = [
    r"\bi am a bot\b",
    r"\bthis (?:comment|post) was (?:posted|left by) a bot\b",
    r"\bthis reply was generated automatically\b",
    r"[\^*]*beep(?:\s+beep)?[\^*]*\s+[\^*]*boop(?:\s+boop)?[\^*]*"
]

BOT_TEXT_RE = re.compile("|".join(BOT_TEXT_PATTERNS), flags=re.IGNORECASE)

URL_RE = re.compile(
    r"""
    (?:
        https?://
        | www\.
        | (?<!@)\b
    )
    (?:[a-zA-Z0-9-]+\.)+
    [a-zA-Z]{2,}
    (?:/[^\s\)\]\}]*)?
    """,
    re.VERBOSE
)

HAS_LETTER_RE = re.compile(r"[A-Za-z]")

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251" 
    "]+",
    flags=re.UNICODE,
)

# r/subreddit patterns
SUBREDDIT_RE = re.compile(r"/?r/[A-Za-z0-9_]+", flags=re.IGNORECASE)

# define relevant sets of tags and words
FINITE_VERB_TAGS = {"VB", "VBD", "VBN", "VBP", "VBZ"}
SUBJECT_TAGS = {"NN", "NNS", "NNP", "NNPS", "PRP"}
SUBORDINATING_CONJ = {"IN"} # tag for subordinating conjunction
COORDINATING_CONJ = {"CC"} # tag for coordinating conjunction

PUNCT = '?!.({[]})-–—"\''
CLOSING_PUNCT = '.!?…'
TRAILING_CLOSERS = set(['"', "'", ')', ']', '}', '”', '’'])

# normalize curly quotes and fancy punctuation
FANCY_TO_ASCII = {
                '“': '"', '”': '"',
                '‘': "'", '’': "'",
                '—': '-', '–': '-',
                '…': '...'
                }

# ----------------------------------------------------------------------------------------
# Lexical Pre-Processing Helper Function
# ----------------------------------------------------------------------------------------
def clean_text(text):
    '''Helper function to clean text by removing urls and other undesirable features.'''

    # remove urls
    text = URL_RE.sub("", text)

    # remove emojis
    text = EMOJI_RE.sub("", text)

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
# Syntactic Pre-Processing Helper Function
# ----------------------------------------------------------------------------------------
def split_sentences(text):
    '''Helper function to split a given utterance into separate sentences'''

    sentence_tokens = sent_tokenize(text)

    return sentence_tokens

def strip_markdown_emphasis(text):
    '''Helper function that remove markdown-style emphasis: *word*, **word**, 
    _word_, __word__'''

    # replace *word* or **word** with word
    text = re.sub(r"(\*{1,2}|_{1,2})(\S.*?\S)\1", r"\2", text)

    # replace strikethrough markdown
    text = re.sub(r'~~', '.', text)
    
    return text

def is_complete_sentence(sentence):
    '''Helper function to determine whether a sentence is complete. Recall that a complete sentence follows these rules:
    -contains at least one subject 
    -contains at least one finite verb
    -ends with appropriate punctuation (.?!) 
    -if it begins with a subordinator, has an independent clause after
    -does not end with a conjunction
    '''

    cleaned = sentence.strip() # removing trailing/leading whitespace
    # account for differences in straight vs. smart quotes
    for f, a in FANCY_TO_ASCII.items():
        cleaned = cleaned.replace(f, a)
    # remove leading/trailing quotes
    cleaned = cleaned.strip('\"')
    cleaned = cleaned.strip('\'')

    # empty string
    if not cleaned:
        return False
    
    # tokenize sentence and tag tokens
    tokens = tokenize(cleaned)
    tags = pos_tag(tokens)

    # ensure length is appropriate
    if len(tokens) < 2:
        return False

    # first letter should be capital
    j = 0
    while j < len(cleaned) and cleaned[j] in PUNCT:
        j += 1
    if j >= len(cleaned):
        return False
    if not cleaned[j].isalpha() or not cleaned[j].isupper():
        return False
        
    # last relevant char must end with proper punctuation
    i = len(cleaned) - 1
    while i > 0 and cleaned[i] in TRAILING_CLOSERS:
        i -= 1
    if i <= 0 or cleaned[i] not in CLOSING_PUNCT:
        return False
    
    # find the first words tag
    first_word = None
    first_tag = None
    for word, tag in tags:
        if word.isalpha():
            first_word = word
            first_tag = tag
            break
    # if first word is subordinating conjunction (including "when"), need independent clause after
    if first_tag in SUBORDINATING_CONJ or first_word == "When":
        if ',' in tokens: # indepdent clause will start after a comma
            comma_index = tokens.index(',')
            post_sub_tags = tags[comma_index+1:]
            # check if independent clause is a complete thought
            has_finite_verb_post_sub = any(tag in FINITE_VERB_TAGS for _, tag in post_sub_tags)
            has_subject_post_sub = any(tag in SUBJECT_TAGS for _, tag in tags)
            if not (has_finite_verb_post_sub and has_subject_post_sub):
                return False
        # if no comma separating clauses
        else:
            noun_count = sum(1 for _, tag in tags if tag in SUBJECT_TAGS)
            verb_count = sum(1 for _, tag in tags if tag in FINITE_VERB_TAGS)
            # edge case for when first word is if
            if first_word == "If" and verb_count < 2:
                return False
            # check for two nouns, if not assume fragment
            if noun_count < 2:
                return False

    # find the last words tag
    last_tag = None
    for word, tag in reversed(tags):
        if word.isalpha():
            last_tag = tag
            break
    # last word cannot be conjunction
    if last_tag in COORDINATING_CONJ:
        return False

    # check if it has finite verb and subject
    has_finite_verb = any(tag in FINITE_VERB_TAGS for _, tag in tags)
    has_subject = any(tag in SUBJECT_TAGS for _, tag in tags)

    return has_finite_verb and has_subject

def clean_tokens_syntactic(text):

    # replace newline characters 
    text = re.sub(r'\n+', '. ', text)

    # replace URLs with "URL" in sentences
    text = URL_RE.sub("URL", text)

    # replace r/subreddit with "this forum" in sentences
    text = SUBREDDIT_RE.sub("this forum", text)

    # strip markdown emphasis
    text = strip_markdown_emphasis(text)

    # remove emojis
    text = EMOJI_RE.sub("", text)

    # handle numbered list items
    text = re.sub(r'\.+\s*(\d+\))', r'. \1', text)
    
    # sentence tokenize text
    sentences = split_sentences(text)

    # remove sentences that do not contain a single letter
    sentences = [s for s in sentences if re.search(HAS_LETTER_RE, s)]

    return sentences

def remove_fragments(sentences):
    ''''''
    complete_sentences = [s for s in sentences if is_complete_sentence(s)]

    return complete_sentences

# ----------------------------------------------------------------------------------------
# Pre-Processing Functions
# ----------------------------------------------------------------------------------------
def filter_df(df):
    '''Helper function that pre-processes a dataframe containing textual social 
    media data by removing deleted/removed utterances, bot utterances, and 
    utterances not containing letters.'''

    # remove deleted/removed utterances
    df = df[~df["raw_text"].str.lower().isin({"[deleted]", "[removed]"})]

    # remove bot authored utterances
    df = df[~df["raw_text"].str.contains(BOT_TEXT_RE, regex=True)]

    # remove utterances without a letter
    df = df[df["raw_text"].str.contains(HAS_LETTER_RE, regex=True)]

    return df

def lexical_preprocessing_df(df):
    '''Function that pre-processes the data in a given dataframe by removing
    deleted/removed utterances, bot utterances, and utterances not containing letters.
    Then, the remaining textual data is tokenized, lemmatized, and cleaned for lexical 
    analysis.'''

    print("Performing lexical preprocessing...\n")
    
    # filter utterances
    df = filter_df(df)

    # final tokenized, lemmatized, and cleaned set
    df["final_lexical_tokens"] = df["raw_text"].apply(clean_tokens_lexical)

    return df

def syntactic_preprocessing_df(df):
    '''Function that pre-processes the data in a given dataframe by removing
    deleted/removed utterances, bot utterances, and utterances not containing letters.
    Then, the remaining textual data is tokenized and cleaned for syntactic analysis.'''

    print("Performing syntactic preprocessing...\n")

    # filter utterances
    df = filter_df(df)

    # final tokenized and cleaned set
    df["candidate_sentences"] = df["raw_text"].apply(clean_tokens_syntactic)

    df["complete_sentences"] = df["candidate_sentences"].apply(remove_fragments)

    return df