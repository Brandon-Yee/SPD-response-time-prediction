# -*- coding: utf-8 -*-
"""
SPD FeatureExtractor
"""
import re
import gensim
import torch
import numpy as np
import pandas as pd

TYPE_FEATURES = ['Event Clearance Description', 'Call Type', 'Initial Call Type', 'Final Call Type']
LOC_FEATURES = ['Precinct', 'Sector', 'Beat']
NUM_FEATURES = ['Priority']

# Event Clearance Description some missing '-', replace with 'none'
class FeatureExtractor():
    """
    OBJECT DESCRIPTION
        Class for processing SPD data features
    FIELDS
        embedding_type - str: Type of embedding to use ('one-hot', 'word2vec')
        word2vec - KeyedVectors: vector embedding vocabulary look up
    """
    def __init__(self, embedding_type, word2vec_path='./GoogleNews-vectors-negative300.bin'):
        """
        BEHAVIOR
            Instantiates the feature extractor for generating embeddings from call types.
        PARAMETERS
            embedding_type - str: Type of embedding to use ('one-hot', 'word2vec')
            word2vec - KeyedVectors: vector embedding vocabulary look up
        RETURNS
            n/a
        """
        #self.embeddings = None
        self.embedding_type = embedding_type
        if embedding_type == 'word2vec':
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            #self.embed_idx = {}
    
    def get_embeddings(self, df):
        """
        BEHAVIOR
            Returns an index dictionary and TorchTensor for DataFrame embeddings
        PARAMETERS
            df - Pandas DataFrame: DataFrame containing strings to embed
        RETURNS
            embed_idx - Dict: lookup table for embedding idx
            embeddings - FloatTensor: # category types x embedding size
        """
        embed_idx = {}
        categories = df.unique()
        if self.embedding_type == 'one-hot':
            for i, category in enumerate(categories):
                embed_idx[category] = i
            embeddings = np.identity(len(categories))
        else:
            embeddings = np.ndarray([len(categories), 300])
            for i, category in enumerate(categories):
                tokenized = self.tokenize(category)
                embed_idx[category] = i
                embeddings[i] = self.vectorize(tokenized)
                if np.isnan(embeddings[i]).any():
                    print(tokenized)
                    print(category)
                
        return embed_idx, torch.from_numpy(embeddings).float()
    
    def transform(self, df, embeddings_dict):
        """
        BEHAVIOR
            Returns a TorchTensor containing the vector representation for the given DataFrame.
        """
        if isinstance(df, pd.Series):
            df = df.to_frame().transpose()
            
        ret = torch.Tensor()
        for feat_type in embeddings_dict:
            indices = embeddings_dict[feat_type][0]
            embeddings = embeddings_dict[feat_type][1]
            idx = [indices[x] for x in df[feat_type].tolist()]
            ret = torch.cat([ret, embeddings[idx]], axis=1)
                
        return ret
            
    def tokenize(self, string):
        """
        BEHAVIOR
            Returns a tokenized bag of words from a string.
        PARAMETERS
            string - str: string to convert
        RETURNS
            tokens - list: bag of words tokens from the string
        """
        tokens = re.split('[\s/]+', re.sub('[-,()]', '', string.lower()))
        return tokens
    
    def vectorize(self, words):
        avg_vector = np.zeros(300)
        count = 0
        for i in words:
            if i in self.word2vec:
                avg_vector += self.word2vec[i]
                count += 1
        if count < 1:
            return self.word2vec['none']
        else:
            return avg_vector/count