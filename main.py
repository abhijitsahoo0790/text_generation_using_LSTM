# -*- coding: utf-8 -*-
"""
Created on 16 July, 2020

@author: Abhijit
"""
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import brown
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import copy
import math
import os
import sys
import traceback
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', 
                    filename='log.txt', filemode='w', level=logging.DEBUG, 
                    datefmt='%Y-%m-%d %H:%M:%S')

START_DELIMITER = "ssttaarrt"
END_DELIMITER = "eenndd"
WINDOW_LENGTH = 20

def fetch_the_corpora_using_NLTK():
    """
    Return the unified corpora from NLTK corpora.

    Returns
    -------
    text : str
        Text data of the corpora.
    """
    corpous_name = "brown"
    status = nltk.download(corpous_name)
    if (status):
        logging.info("Downloaded Brown corpus")
        mdetok = TreebankWordDetokenizer()
        brown_natural = [mdetok.detokenize(' '.join(sent).replace('``', '"').replace("''", '"').replace('`', "'").split())  for sent in brown.sents()]
        logging.info("Processed Brown corpus as text")
    else:
        logging.error("Couldn't download the "+ corpous_name+" corpus")
        
    return brown_natural

def enumerate_text_using_word_enum_dict(unified_corpora, word_enum_dict):
    """
    Enumerate the complete text in corpous using word_enum_dict

    Parameters
    ----------
    unified_corpora : TYPE
        DESCRIPTION.
    word_enum_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    complete_text_enumerated : TYPE
        DESCRIPTION.

    """
    complete_text = " ".join([START_DELIMITER+" "+item+" "+END_DELIMITER for item in unified_corpora])
    complete_text_processed = re.sub(' +', ' ', re.sub('[^A-Za-z ]+', ' ',complete_text.lower())).strip()
    complete_text_enumerated =  [word_enum_dict[item] for item in complete_text_processed.split(" ") if item in word_enum_dict]   
    return complete_text_enumerated

def enumerate_unique_words(unified_corpora):
    """
    Enumerate unique words and return its dictionary and reversed-dictionary

    Parameters
    ----------
    unified_corpora : list of str
        The text corpora as a list of words 

    Returns
    -------
    word_enum_dict
        word as key and its integer enumeration as the value.
    reversed_word_enum_dict
        word as value and its integer enumeration as the key.
    """

    """
    Join all sentences, remove special characters except Space, split all 
    words, take set for unique words, convert it to list, remove None values using filter
    """
    unique_words = list(filter(None, list(set(re.sub('[^A-Za-z ]+', ' ', (''.join(unified_corpora).lower())).split(" ")))))
    unique_words = unique_words + [START_DELIMITER, END_DELIMITER]
    #enumerate unique words
    word_enum_dict = {v:k for k,v in enumerate(unique_words)}
    reversed_word_enum_dict = {k:v for k,v in enumerate(unique_words)}
    return [word_enum_dict, reversed_word_enum_dict]

def generate_sequence_data_for_LSTM(complete_text_enumerated):
    """
    Generate pattern sequences of length as specified by WINDOW_LENGTH and 
    also generate target of the patterns generated.

    Parameters
    ----------
    complete_text_enumerated : list of int
        Enumerated text sequence.

    Returns
    -------
    X
        Reshaped pattern sequences for LSTM input .
    y
        Target for each generated patterns.
    """
    pattern_sequence = []
    pattern_targets = []
    for i in range(0, len(complete_text_enumerated)-WINDOW_LENGTH):
        temp_pattern = complete_text_enumerated[i:i+WINDOW_LENGTH]
        temp_pattern_target = complete_text_enumerated[i+WINDOW_LENGTH]
        pattern_sequence.append(temp_pattern)
        pattern_targets.append(temp_pattern_target)
    num_patterns = len(pattern_sequence)
    X = np.reshape(pattern_sequence, (num_patterns, WINDOW_LENGTH, 1))
    y = np_utils.to_categorical(pattern_targets)
    return [X, y]


if __name__ == "__main__":
    logging.info("Fetching text corpus...")
    unified_corpora = fetch_the_corpora_using_NLTK()    
    logging.info("Fetched text corpus")
    
    # Enumerate unique words
    [word_enum_dict, reversed_word_enum_dict] = enumerate_unique_words(unified_corpora)
    # Enumerate text using word_enum_dict
    complete_text_enumerated = enumerate_text_using_word_enum_dict(unified_corpora, word_enum_dict)
    # generate sequence data for training LSTM
    [X, y] = generate_sequence_data_for_LSTM(complete_text_enumerated)