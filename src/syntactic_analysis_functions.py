# ----------------------------------------------------------------------------------------
# This code is designed to compute measures of syntactic complexity from Reddit data
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# general imports
import numpy as np
from tqdm import tqdm
import nltk

# syntactic specific imports
from nltk import pos_tag
from nltk.corpus import treebank
from nltk.tree import ParentedTree

# import spacy nlp model
import spacy
nlp = spacy.load("en_core_web_sm")
import stanza

stanza_parser = stanza.Pipeline(
    "en",
    processors="tokenize,pos,constituency",
    use_gpu=False,
    download_method=None
)

# import data processing functions
from data_preprocessing import is_complete_sentence, clean_tokens_lexical, clean_tokens_syntactic, remove_fragments

# ----------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------
def create_parented_tree(complete_sent):
    '''Helper function to create a parented tree for a valid sentence'''
    
    doc = stanza_parser(complete_sent)
    stanza_tree = doc.sentences[0].constituency
    parented_tree = ParentedTree.fromstring(str(stanza_tree))
    
    return parented_tree

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
    '''Helper function to count the number of clauses in a complete sentence.'''

    clause_count = 0

    # only consider complete sentences
    if not is_complete_sentence(complete_sent):
        return 0
    
    if t_unit_count is None:
        t_unit_count = count_t_units(complete_sent, ptree=ptree)

    # create a dependency tree
    if ptree is None:
        ptree = create_parented_tree(complete_sent)
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

def compute_syntactic_vals(df):
    '''Function to compute the syntactic metrics for each utterance in a dataframe.'''

    # convert the text into a list of candidate sentences
    df = syntactic_preprocessing_df(df)
    num_utterances = len(df)

    # initialize lists to store results
    fragment_ratio_list = []
    avg_t_units_list = []
    clause_t_unit_ratio_list = []
    mltu_list = []

    # compute fragment ratio
    for candidate, complete in tqdm(zip(df["candidate_sentences"], df["complete_sentences"]), 
                                    total=num_utterances, desc="Computing fragment ratios"):
        if not candidate:
            fragment_ratio_list.append(np.nan)
        else:
            fragment_ratio_list.append(fragment_ratio(candidate, complete))

    # compute average t_units per sentence, clause to t-unit ratio, and mltu values
    for sentences in tqdm(df["complete_sentences"], 
                          total=num_utterances, desc="Computing remaining values per sentence"):
        if not sentences:
            avg_t_units_list.append(np.nan)
            clause_t_unit_ratio_list.append(np.nan)
            mltu_list.append(np.nan)
        else:
            avg_t_units_list.append(avg_t_units_per_sentence(sentences))
            clause_t_unit_ratio_list.append(clause_t_unit_ratio(sentences))
            mltu_list.append(mltu(sentences))
        
    # store all values in dataframe
    df["fragment_ratio"] = fragment_ratio_list
    df["avg_t_units"] = avg_t_units_list
    df["clause_to_t_unit_ratio"] = clause_t_unit_ratio_list
    df["mltu"] = mltu_list

    return df