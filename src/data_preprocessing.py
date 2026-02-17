# ----------------------------------------------------------------------------------------
# This code is designed to download and pre-process the Reddit data to be used in analysis
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# general imports
import pandas as pd
from datetime import datetime
import re
import hashlib
import json
import os

# set up nltk tokenizers
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import pos_tag
from nltk.corpus import wordnet

# ----------------------------------------------------------------------------------------
# Format Data
# ----------------------------------------------------------------------------------------
BATCH_SIZE = 100

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

def corpus_to_df_batches(corpus, batch_size=BATCH_SIZE):
    '''Function to convert the convokit  corpus to a pandas dataframe structure in chunks
    so as to reduce memory capacity demanded of the cluster.'''

    rows = []
    for utt in corpus.iter_utterances():
        # only consider utterances with timestamps and text
        if hasattr(utt, "timestamp") and utt.text:
            # convert timestamp from seconds since 1/1/1970 to datetime
            t = datetime.fromtimestamp(int(utt.timestamp))

            rows.append({
                "utterance_id": utt.id,
                "speaker_id": utt.speaker.id,
                "raw_text": utt.text,
                "timestamp": t
            })

            if len(rows) >= batch_size:
                yield pd.DataFrame(rows)
                rows = []

    if rows:
        yield pd.DataFrame(rows)

def _speaker_shard_index(speaker_id, num_shards):
    '''Deterministically assign a speaker to a shard index in [0, num_shards).'''
    digest = hashlib.md5(str(speaker_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards

def _extract_utterance_fields_json(obj):
    '''Extract utterance fields from a Convokit JSONL row.'''

    utt_id = obj.get("id", obj.get("utterance_id"))
    raw_text = obj.get("text", obj.get("raw_text"))
    timestamp = obj.get("timestamp")

    speaker = obj.get("speaker", obj.get("speaker_id"))
    if isinstance(speaker, dict):
        speaker_id = speaker.get("id", speaker.get("speaker_id"))
    else:
        speaker_id = speaker

    return utt_id, speaker_id, raw_text, timestamp

def corpus_longest_posts_batches_from_jsonl(corpus_dir, batch_size=BATCH_SIZE):
    '''Stream utterances.jsonl directly, keep longest valid post per speaker globally,
    then yield rows in batches.'''

    utt_path = os.path.join(corpus_dir, "utterances.jsonl")
    if not os.path.isfile(utt_path):
        raise FileNotFoundError(f"Could not find utterances.jsonl at {utt_path}")

    best_by_speaker = {}
    counts_by_speaker = {}

    # Pass 1: compute per-speaker count and longest utterance metadata.
    with open(utt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            utt_id, speaker_id, raw_text, timestamp = _extract_utterance_fields_json(obj)
            if speaker_id is None or not raw_text or timestamp is None or utt_id is None:
                continue

            lower_text = raw_text.lower()
            if lower_text in {"[deleted]", "[removed]"}:
                continue
            if BOT_TEXT_RE.search(raw_text):
                continue
            if not HAS_LETTER_RE.search(raw_text):
                continue

            counts_by_speaker[speaker_id] = counts_by_speaker.get(speaker_id, 0) + 1

            post_length = len(raw_text)
            prev = best_by_speaker.get(speaker_id)
            if prev is None or post_length > prev["post_length"]:
                try:
                    dt = datetime.fromtimestamp(int(timestamp))
                except (TypeError, ValueError, OSError):
                    continue
                best_by_speaker[speaker_id] = {
                    "utterance_id": utt_id,
                    "timestamp": dt,
                    "post_length": post_length,
                }

    selected_utterance_to_speaker = {
        row["utterance_id"]: speaker_id for speaker_id, row in best_by_speaker.items()
    }

    # Pass 2: emit only selected utterances in batches.
    rows = []
    with open(utt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            utt_id, _, raw_text, _ = _extract_utterance_fields_json(obj)
            speaker_id = selected_utterance_to_speaker.get(utt_id)
            if speaker_id is None:
                continue

            rows.append({
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "raw_text": raw_text,
                "timestamp": best_by_speaker[speaker_id]["timestamp"],
                "num_utterances_by_speaker": counts_by_speaker[speaker_id],
            })
            if len(rows) >= batch_size:
                yield pd.DataFrame(rows)
                rows = []

    if rows:
        yield pd.DataFrame(rows)

# ----------------------------------------------------------------------------------------
# Global Variables for DF-Level Cleaning
# ----------------------------------------------------------------------------------------
# various bot text flags
BOT_TEXT_PATTERNS = [
    r"\bi am a bot\b",
    r"\bthis (?:comment|post) was (?:posted|left by) a bot\b",
    r"\bthis reply was generated automatically\b",
    r"[\^*]*beep(?:\s+beep)?[\^*]*\s+[\^*]*boop(?:\s+boop)?[\^*]*"
]
BOT_TEXT_RE = re.compile("|".join(BOT_TEXT_PATTERNS), flags=re.IGNORECASE)

# various https, www, and raw domain URL patterns
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

# emoji unicode patterns
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols and pictographs
    "\U0001F680-\U0001F6FF"  # transport and map symbols
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
    '''Function that pre-processes a dataframe containing textual social 
    media data by removing deleted/removed utterances, bot utterances, and 
    utterances not containing letters. Retains only the longest post per unique
    user while recording the number of utterances they authored in a new column.'''

    print("Removing spam utterances...")

    # remove deleted/removed utterances
    df = df[~df["raw_text"].str.lower().isin({"[deleted]", "[removed]"})]

    # remove bot authored utterances
    df = df[~df["raw_text"].str.contains(BOT_TEXT_RE, regex=True)]

    # remove utterances without a letter
    df = df[df["raw_text"].str.contains(HAS_LETTER_RE, regex=True)]

    print("Selecting longest utterances...")

    # compute number of posts per speaker
    df["num_utterances_by_speaker"] = df.groupby("speaker_id")["raw_text"].transform("count")

    # compute post length (character count)
    df["post_length"] = df["raw_text"].str.len()

    # retain only the longest post per speaker
    df = df.loc[df.groupby("speaker_id")["post_length"].idxmax()]

    # drop helper column
    df = df.drop(columns=["post_length"])

    return df

def corpus_longest_posts_batches(corpus, batch_size=BATCH_SIZE, num_shards=1, shard_index=0):
    '''Stream corpus once, keep only the longest valid post per speaker globally,
    then yield those rows in batches.'''

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    print("Extracting longest post per user...")

    # keep metadata only in pass 1 to avoid storing many full texts in memory.
    best_by_speaker = {}
    counts_by_speaker = {}

    for utt in corpus.iter_utterances():
        # only consider utterances with timestamps and text
        if not hasattr(utt, "timestamp") or not utt.text:
            continue

        raw_text = utt.text
        lower_text = raw_text.lower()

        # mirror filter_df cleaning criteria during the global pass
        if lower_text in {"[deleted]", "[removed]"}:
            continue
        if BOT_TEXT_RE.search(raw_text):
            continue
        if not HAS_LETTER_RE.search(raw_text):
            continue

        speaker_id = utt.speaker.id
        if _speaker_shard_index(speaker_id, num_shards) != shard_index:
            continue
        counts_by_speaker[speaker_id] = counts_by_speaker.get(speaker_id, 0) + 1

        post_length = len(raw_text)
        prev = best_by_speaker.get(speaker_id)
        # keep first max-length post encountered (matches idxmax tie behavior)
        if prev is None or post_length > prev["post_length"]:
            best_by_speaker[speaker_id] = {
                "utterance_id": utt.id,
                "timestamp": datetime.fromtimestamp(int(utt.timestamp)),
                "post_length": post_length,
            }

    selected_utterance_to_speaker = {
        row["utterance_id"]: speaker_id for speaker_id, row in best_by_speaker.items()
    }

    rows = []
    # pass 2: recover raw text only for selected utterances and emit in batches.
    for utt in corpus.iter_utterances():
        speaker_id = selected_utterance_to_speaker.get(utt.id)
        if speaker_id is None:
            continue

        row = best_by_speaker[speaker_id]
        rows.append({
            "utterance_id": utt.id,
            "speaker_id": speaker_id,
            "raw_text": utt.text,
            "timestamp": row["timestamp"],
            "num_utterances_by_speaker": counts_by_speaker[speaker_id],
        })
        if len(rows) >= batch_size:
            yield pd.DataFrame(rows)
            rows = []

    if rows:
        yield pd.DataFrame(rows)

def lexical_preprocessing_df(df):
    '''Function that pre-processes the data in a given dataframe by removing
    deleted/removed utterances, bot utterances, and utterances not containing letters.
    Then, the remaining textual data is tokenized, lemmatized, and cleaned for lexical 
    analysis.'''

    print("Performing lexical preprocessing...\n")

    # final tokenized, lemmatized, and cleaned set
    df["final_lexical_tokens"] = df["raw_text"].apply(clean_tokens_lexical)

    return df

def syntactic_preprocessing_df(df):
    '''Function that pre-processes the data in a given dataframe by removing
    deleted/removed utterances, bot utterances, and utterances not containing letters.
    Then, the remaining textual data is tokenized and cleaned for syntactic analysis.'''

    print("Performing syntactic preprocessing...\n")

    # final tokenized and cleaned set
    df["candidate_sentences"] = df["raw_text"].apply(clean_tokens_syntactic)

    df["complete_sentences"] = df["candidate_sentences"].apply(remove_fragments)

    return df
