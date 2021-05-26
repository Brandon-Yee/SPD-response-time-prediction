# -*- coding: utf-8 -*-
"""
SPD FeatureExtractor
"""
import gensim
import pandas as pd

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
        self.embedding_type = embedding_type
        if embedding_type == 'word2vec':
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            
    def get_embeddings(self, df):
        """
        BEHAVIOR
            Returns a Pandas DataFrame containing the embeddings produced from the given Dataframe.
        PARAMETERS
            df - Pandas DataFrame: DataFrame containing strings to embed
        RETURNS
            embeddings - Pandas DataFrame: DataFrame containing the embeddings
        """
        if self.embedding_type == 'one-hot':
            embeddings = pd.get_dummies(df, prefix='Init_Call_Type')
        else:
            tokenize_df = df.apply(lambda x: self.tokenize(x))
            embeddings = self.word2vec[tokenize_df]
        return embeddings
    
    def tokenize(self, string):
        """
        BEHAVIOR
            Returns a tokenized bag of words from a string.
        PARAMETERS
            string - str: string to convert
        RETURNS
            tokens - list: bag of words tokens from the string
        """
        pass