# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import pandas as pd
import numpy as np
import nltk
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
    # fetch all the corpora
    text = ""
    
    
    return text


if __name__ == "__main__":
    unified_corpora = fetch_the_corpora_using_NLTK()
    