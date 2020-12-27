'''
	Use thie file to generate result.
'''
import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from bert4keras.backend import keras, set_gelu
from keras.layers import *
from keras import backend as K
from sklearn import metrics
import gensim
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from utils import save_variable, load_variavle, data_generator_forward, data_generator_backward
from model import my_HAN_model
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


MAX_SENT_LENGTH = 50
MAX_SENTS = 350
EMBEDDING_DIM = 100
batch_size = 64

test_data = pd.read_csv("./data/Test_B.csv")
test = test_data[['SessionId']].drop_duplicates().reset_index(drop=True)

def get_text(row):
    return list(row['Role']+":"+row['Text'])
tmp = test_data.groupby('SessionId').apply(get_text).rename('text')
test = pd.merge(test, tmp, on=['SessionId'], how='left', copy=False)
test['HighRiskFlag'] = 0

vocab = load_variavle('./model/vocab_set.pkl')
w2v_model = gensim.models.Word2Vec.load("./model/w2v.model")
# 预训练的词向量中没有出现的词用0向量表示
embedding_matrix = np.zeros((len(vocab) + 1, 100))
for word, i in vocab.items():
    try:
        embedding_vector = w2v_model[str(word)]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue

# 5折结果
probs = [0 for x in range(5)]

for fold in range(5):

    K.clear_session()
    md = my_HAN_model(MAX_SENT_LENGTH = MAX_SENT_LENGTH, MAX_SENTS=MAX_SENTS, embedding_matrix=embedding_matrix,vocab=vocab)
    model = md.create_model()
    
    test_generator = data_generator_forward(test.values, batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS, train=False)

    model.load_weights(r'./model/best_model{}.weights'.format(fold))
    probs[fold] = model.predict_generator(test_generator.forpredict(),
        steps=len(test_generator))[:,0]

y_test = np.mean(probs, axis=0)
np.savetxt("B_HAN_resampl_5_probs.txt", probs)
sub1 = test.copy()
sub1['Probability'] = y_test
sub1[['SessionId','Probability']].to_csv('./result/B_HAN_resampl_5.csv', index=None)

probs = [0 for x in range(5)]

# 5折结果

probs = [0 for x in range(5)]
for fold in range(5):

    K.clear_session()
    md = my_HAN_model(MAX_SENT_LENGTH = MAX_SENT_LENGTH, MAX_SENTS=MAX_SENTS, embedding_matrix=embedding_matrix,vocab=vocab)
    model = md.create_model()
    
    test_generator = data_generator_backward(test.values, batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS, train=False)

    model.load_weights(r'./model/best_han_model_rump_{}.weights'.format(fold))
    probs[fold] = model.predict_generator(test_generator.forpredict(),
        steps=len(test_generator))[:,0]

y_test = np.mean(probs, axis=0)
sub2 = test.copy()
sub2['Probability'] = y_test
sub2[['SessionId','Probability']].to_csv('./result/B_HAN_rump.csv', index=None)

# merge
sub_merge = sub1.copy()
sub1['rank'] = 1/sub1['Probability'].rank(ascending=False)
# print(sub1)
sub2['rank'] = 1/sub2['Probability'].rank(ascending=False)
# print(sub2)
sub_merge['Probability'] = sub1['rank'] + sub2['rank']
sub_merge[['SessionId','Probability']].to_csv("./result/sub_merge_all.csv", index=None)