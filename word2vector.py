import re
import sys
import nltk
from gensim.models import Word2Vec
import os.path
from os import path 
import pandas as pd

def reviews(file_path):
    file_path_neg = file_path + r"/neg.txt"
    file_path_pos = file_path + r"/pos.txt"
    with open(file_path_neg, 'r') as file:
        neg_comments = file.readlines()
    with open(file_path_pos, 'r') as file:
        pos_comments = file.readlines()
        
    return neg_comments, pos_comments

class AmazonCorpus:
    def __init__(self, file_path):
        self.neg_comments, self.pos_comments = reviews(file_path)
        self.comments = self.neg_comments + self.pos_comments

    def remove_characters(self):
        comments = []
        for line in self.comments:
            special_char = """[.,!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n]+"""
            comments_special_char = re.sub(special_char, ' ', line.lower())
            comments.append(comments_special_char)
        return comments
    
    def tokenization(self,comments):
        
        tokens = []
        for line in comments:
            token =  line.split()
            tokens.append(token)
        return tokens
    
    def train_word2vec(self, tokenization):
        model = Word2Vec(tokenization, min_count = 5, window = 5, workers = 4)
        self.word2vec_model = model

    def model_save(self, model_file):
        self.word2vec_model.save(model_file)

    def get_model(self):
        return self.word2vec_model

def main():
    model_file = r'data/word2vec.pkl'   
    file_path = sys.argv[1]
    Reviews = AmazonCorpus(file_path)
    comments = Reviews.remove_characters()
    tokenization = Reviews.tokenization(comments)
    word2vec = Reviews.train_word2vec(tokenization)
    model_save = Reviews.model_save(model_file)
    model = Reviews.get_model()
    print('Similar words for good: ') 
    print(pd.Series(dict(model.wv.most_similar('good', topn=20))))
    print('Similar words for bad:') 
    print(pd.Series(dict(model.wv.most_similar('bad', topn=20))))
    print(word2vec)

if __name__ == '__main__':
    main()
