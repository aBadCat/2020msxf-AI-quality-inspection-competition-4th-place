'''
	Pre training word2vec with preliminary training set and test set
	Get vocab and embed
'''
import numpy as np
import pandas as pd
import jieba
import gensim
from keras.preprocessing.text import Tokenizer
from utils import save_variable, load_variavle

data = pd.read_csv("./data/Train.csv")
test_data = pd.read_csv("./data/Test_A.csv")
train = data[['SessionId','HighRiskFlag']].drop_duplicates().reset_index(drop=True)
test = test_data[['SessionId']].drop_duplicates().reset_index(drop=True)
train_labels = train.HighRiskFlag.values

# with no stop_words
stop_words = []

vocab_set_path = './model/vocab_set.pkl'
def get_text(row):
    return " ".join(row['Role']+":"+row['Text'])
tmp = data.groupby('SessionId').apply(get_text).rename('text')
train = pd.merge(train, tmp, on=['SessionId'], how='left', copy=False)
tmp = test_data.groupby('SessionId').apply(get_text).rename('text')
test = pd.merge(test, tmp, on=['SessionId'], how='left', copy=False)
test['HighRiskFlag'] = 0

# cut words
def cut_text(sentence):
    tokens = jieba.lcut(sentence)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

train['text'] = train['text'].map(lambda x: cut_text(x))
test['text'] = test['text'].map(lambda x: cut_text(x))

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(np.r_[train['text'].values,test['text'].values])
vocab = tokenizer.word_index

# save vocab
save_variable(vocab,vocab_set_path)

vector_size = 100
model = gensim.models.Word2Vec(size=vector_size, window=2, min_count=1, workers=5, sg=0, iter=20, seed=2020)
model.build_vocab(np.r_[train['text'].values,test['text'].values])
model.train(np.r_[train['text'].values,test['text'].values], total_examples=model.corpus_count, epochs=model.iter)
model.save("./model/w2v.model")
word2vec_save = './model/word2vec_model.txt'
model.wv.save_word2vec_format(word2vec_save, binary=False)

