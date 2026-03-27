# ----------------------------------------------------------------------------------------
# This code is designed to compute measures of syntactic complexity from Reddit data
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# general imports
import multiprocessing
import numpy as np
from tqdm import tqdm
import nltk

# syntactic specific imports
from nltk import pos_tag
from nltk.corpus import treebank
from nltk.tree import ParentedTree

import stanza

# Stanza parser is NOT initialised at import time so that worker processes can
# be forked cleanly before the model is loaded.  Call _init_stanza() (or let
# create_parented_tree do so lazily) before parsing.
stanza_parser = None

# Guardrails to avoid parser memory blow-ups on pathological long sentences.
MAX_PARSE_CHARS = 1200
MAX_PARSE_TOKENS = 120

# import data processing functions
from data_preprocessing import is_complete_sentence, clean_tokens_lexical, clean_tokens_syntactic, remove_fragments


def _init_stanza():
    """Initialise (or return the already-cached) Stanza pipeline in the
    current process.  Safe to call multiple times; only loads the model once."""
    global stanza_parser
    if stanza_parser is None:
        stanza_parser = stanza.Pipeline(
            "en",
            processors="tokenize,pos,constituency",
            use_gpu=False,
            download_method=None,
        )

# ----------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------
def create_parented_tree(complete_sent):
    '''Helper function to create a parented tree for a valid sentence'''

    text = complete_sent.strip()
    if not text:
        return None

    # Very long sentences can cause constituency parsing to consume extreme memory.
    if len(text) > MAX_PARSE_CHARS or len(text.split()) > MAX_PARSE_TOKENS:
        return None

    _init_stanza()
    try:
        doc = stanza_parser(text)
        if not doc.sentences:
            return None
        stanza_tree = doc.sentences[0].constituency
        return ParentedTree.fromstring(str(stanza_tree))
    except Exception:
        return None

def count_t_units(complete_sent, ptree=None):
    '''Helper function that returns the number of t-units in a sentence'''

    t_unit_count = 0
    is_question = False
    counted_s_label = False
    has_nested_sq_label = False
    parent_label = None
    to_decremented = False
    
    # create a dependency tree
    if ptree is None:
        ptree = create_parented_tree(complete_sent)
    if ptree is None:
        return 1

    # iterated through parented subtrees
    for subtree in ptree.subtrees():

        # extract relevant labels
        label = subtree.label()
        if subtree.parent():
            parent_label = subtree.parent().label()

        # flag if the sentence is a question and thus has different rules
        if label in {"SQ", "SBARQ"}:
             is_question = True
             # if we've counted a preceding S label decrement
             if counted_s_label:
                t_unit_count -= 1
                counted_s_label = False

        # logic if sentence is a question
        if is_question:
            if label == "SQ":
                t_unit_count += 1
                # if nested SQ label, flag
                if parent_label == "SQ":
                    has_nested_sq_label = True

        # logic when sentence is not a question
        else:
            # subtract occurences when "to" is considered a new subject
            if label == "TO" and not to_decremented:
                t_unit_count -= 1
                to_decremented = True
                    
            # check for subjects in regular sentences
            if label == "S":
                # if subject belongs to subordinate clause, ignore
                if parent_label == "SBAR":
                    continue
                # otherwise increment
                counted_s_label = True
                t_unit_count += 1
    
    # ignore duplicated subject labels
    if t_unit_count > 1 and not is_question:
        t_unit_count -= 1
    if has_nested_sq_label:
        t_unit_count -= 1

    # adjust for inappropriate decrements
    if t_unit_count == 0:
        if to_decremented:
            t_unit_count += 1

    # heuristic for special constructions
    if t_unit_count == 0:
        return 1

    return t_unit_count

def extract_t_units(complete_sent, ptree=None):
    '''Helper function that returns a list of the t-units in a complete sentence.'''
    
    t_units = []
    is_question = False
    
    # create a dependency tree
    if ptree is None:
        ptree = create_parented_tree(complete_sent)
    if ptree is None:
        return [complete_sent]
    
    # flag if the sentence is a question
    for subtree in ptree.subtrees():
        if subtree.label() in {"SQ", "SBARQ"}:
            is_question = True
            break
    
    # extract t-units based on sentence type
    if is_question:
        # for questions, extract SQ constituents
        sq_found = False
        for subtree in ptree.subtrees():
            if subtree.label() == "SQ":
                parent_label = subtree.parent().label() if subtree.parent() else None
                # skip nested SQ labels
                if parent_label == "SQ":
                    continue
                # for top-level SQ, check if it has coordinated SQ children
                child_sqs = [child for child in subtree if hasattr(child, 'label') and child.label() == "SQ"]
                if child_sqs:
                    # has coordinated SQ children, extract those instead
                    for child_sq in child_sqs:
                        t_unit_text = " ".join(child_sq.leaves())
                        t_units.append(t_unit_text)
                        sq_found = True
                else:
                    # no coordinated children, extract this SQ
                    t_unit_text = " ".join(subtree.leaves())
                    t_units.append(t_unit_text)
                    sq_found = True
        
        # if no SQ found, fall back to extracting the whole question
        if not sq_found:
            t_units.append(complete_sent)
            
    else:
        # for declarative sentences, extract S constituents that are direct children of root or coordinated
        for subtree in ptree.subtrees():
            if subtree.label() == "S":
                parent_label = subtree.parent().label() if subtree.parent() else None
                # skip if subject belongs to subordinate clause
                if parent_label == "SBAR":
                    continue
                # skip the top-most S that contains everything
                if parent_label in {None, "ROOT"} and len([s for s in ptree.subtrees() if s.label() == "S"]) > 1:
                    continue
                t_unit_text = " ".join(subtree.leaves())
                t_units.append(t_unit_text)
    
    # remove duplicates while preserving order
    seen = set()
    unique_t_units = []
    for t_unit in t_units:
        if t_unit not in seen:
            seen.add(t_unit)
            unique_t_units.append(t_unit)
    
    # filter out T-units that are only infinitive clauses (start with "to" and have no subject)
    # keep T-units that have a subject before "to" (e.g., "I want to leave")
    filtered_t_units = []
    for t_unit in unique_t_units:
        words = t_unit.strip().split()
        # if it starts with "to", likely an infinitive clause fragment - remove it, unless it's the only t-unit
        if words and words[0].lower() == "to" and len(unique_t_units) > 1:
            continue
        filtered_t_units.append(t_unit)
    
    # heuristic: if no t-units found but it's a complete sentence, return the whole sentence
    if len(filtered_t_units) == 0:
        return [complete_sent]
    
    return filtered_t_units

def count_clauses(complete_sent, ptree=None, t_unit_count=None):
    '''Helper function to count the number of clauses in a complete sentence.
    Assumes complete_sent has already been validated by remove_fragments /
    is_complete_sentence — the caller is responsible for that pre-filter.'''

    clause_count = 0

    if t_unit_count is None:
        t_unit_count = count_t_units(complete_sent, ptree=ptree)

    # create a dependency tree
    if ptree is None:
        ptree = create_parented_tree(complete_sent)
    if ptree is None:
        return t_unit_count
    # print(TreePrettyPrinter(ptree))

    # iterated through parented subtrees
    for subtree in ptree.subtrees():
        # if subject belongs to subordinate clause, increment 
        if subtree.label() == "SBAR":
            clause_count += 1

    return clause_count + t_unit_count

def t_unit_length(t_unit):
    '''Helper function that determines the number of tokens in a given t-unit, 
    a.k.a. the t-unit length '''

    tokens = clean_tokens_lexical(t_unit)
    
    return len(tokens)

def compute_sentence_stats(complete_sent):
    '''Parses a sentence once and returns the syntactic counts needed by all metrics.'''

    ptree = create_parented_tree(complete_sent)
    t_units = extract_t_units(complete_sent, ptree=ptree)
    t_count = count_t_units(complete_sent, ptree=ptree)
    clause_count = count_clauses(complete_sent, ptree=ptree, t_unit_count=t_count)

    return t_count, clause_count, t_units

# ----------------------------------------------------------------------------------------
# Lexical Analysis Functions
# ----------------------------------------------------------------------------------------
def fragment_ratio(candidate_sentences, complete_sentences):
    '''Function to determine the ratio of fragments to lines in a given text'''

    # compute the total number of candidates
    total = len(candidate_sentences)
    if total == 0:
        return np.nan

    # find the total number of fragments
    num_fragments = total - len(complete_sentences)

    return num_fragments / total

def avg_t_units_per_sentence(complete_sentences):
    '''Function that, given an utterance, computes the number of t_units per sentence 
    and returns the average across all sentences.'''

    # find the number of sentences
    num_sentences = len(complete_sentences)

    # find the number of t_units
    t_units_per_sent = [count_t_units(sent) for sent in complete_sentences]
    num_t_units = sum(t_units_per_sent)

    return num_t_units / num_sentences

def clause_t_unit_ratio(complete_sentences):

    # find the total number of clauses in the utterance
    num_clauses_per_sent = [count_clauses(sent) for sent in complete_sentences]
    total_clauses = sum(num_clauses_per_sent)

    # find the total number of t_units in the utterance
    num_t_units_per_sent = [count_t_units(sent) for sent in complete_sentences]
    total_t_units = sum(num_t_units_per_sent)

    return total_clauses / total_t_units

def mltu(complete_sentences):
    '''Computes the Mean Length of a T-Unit (MLTU) in a particular utterance.'''

    # extract the t_units
    t_units = []
    for sent in complete_sentences:
        t_units.append(extract_t_units(sent))
    # flatten the t_unit list
    t_units = [item for sublist in t_units for item in sublist]

    lengths = []
    # determine the length of each t-unit
    for unit in t_units:
        lengths.append(t_unit_length(unit))

    return np.mean(lengths)

def _worker_process_text(text):
    '''Top-level worker function — must be defined at module scope so that it
    is picklable by multiprocessing.  Processes a single raw_text string and
    returns a tuple (fragment_ratio, avg_t_units, clause_to_t_unit_ratio, mltu).

    stanza_parser must already be initialised in the calling process (the Pool
    initializer _init_stanza handles this for worker processes).
    '''
    candidate_sentences = clean_tokens_syntactic(text)
    complete_sentences = remove_fragments(candidate_sentences)

    frag_r = (
        fragment_ratio(candidate_sentences, complete_sentences)
        if candidate_sentences
        else np.nan
    )

    if not complete_sentences:
        return frag_r, np.nan, np.nan, np.nan

    total_t_units = 0
    total_clauses = 0
    t_unit_lengths = []

    for sent in complete_sentences:
        t_count, clause_count, t_units = compute_sentence_stats(sent)
        total_t_units += t_count
        total_clauses += clause_count
        for unit in t_units:
            t_unit_lengths.append(t_unit_length(unit))

    avg_t = total_t_units / len(complete_sentences)
    c_t_r = total_clauses / total_t_units if total_t_units else np.nan
    mltu_val = float(np.mean(t_unit_lengths)) if t_unit_lengths else np.nan

    return frag_r, avg_t, c_t_r, mltu_val


def compute_syntactic_vals(df, n_workers=1):
    '''Compute syntactic metrics for each utterance in a dataframe.

    Parameters
    ----------
    df        : DataFrame with a "raw_text" column.
    n_workers : Number of parallel worker processes.  Defaults to 1
                (single-threaded).  Set to the number of CPUs allocated in
                your SLURM script to enable concurrent Stanza parsing.
    '''
    texts = df["raw_text"].tolist()
    num_utterances = len(texts)

    if n_workers > 1:
        # Fork before any Stanza model is loaded in the parent, then let each
        # worker initialise its own pipeline via the pool initializer.
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_workers, initializer=_init_stanza) as pool:
            results = list(
                tqdm(
                    pool.imap(_worker_process_text, texts, chunksize=10),
                    total=num_utterances,
                    desc="Computing syntactic values",
                )
            )
    else:
        _init_stanza()
        results = [
            _worker_process_text(text)
            for text in tqdm(texts, total=num_utterances, desc="Computing syntactic values")
        ]

    if results:
        frag_ratios, avg_t_units, c_t_ratios, mltus = zip(*results)
    else:
        frag_ratios, avg_t_units, c_t_ratios, mltus = [], [], [], []

    df["fragment_ratio"] = list(frag_ratios)
    df["avg_t_units"] = list(avg_t_units)
    df["clause_to_t_unit_ratio"] = list(c_t_ratios)
    df["mltu"] = list(mltus)

    return df
