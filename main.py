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

def enumerate_text_wrt_word_enum_dict(unified_corpora, word_enum_dict):
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
    complete_text = " ".join(unified_corpora)
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
    #enumerate unique words
    word_enum_dict = {v:k for k,v in enumerate(unique_words)}
    reversed_word_enum_dict = {k:v for k,v in enumerate(unique_words)}
    return [word_enum_dict, reversed_word_enum_dict]

if __name__ == "__main__":
    logging.info("Fetching text corpus...")
    unified_corpora = fetch_the_corpora_using_NLTK()    
    logging.info("Fetched text corpus")
        
    # Enumerate unique words
    [word_enum_dict, word_enum_dict] = enumerate_unique_words(unified_corpora)
    # Enumerate text using word_enum_dict
    complete_text_enumerated = enumerate_text_wrt_word_enum_dict(unified_corpora, word_enum_dict)