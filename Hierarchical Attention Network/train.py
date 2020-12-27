import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from bert4keras.backend import keras, set_gelu
from keras.layers import *
from keras import backend as K
from sklearn import metrics
import jieba
import gensim
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from utils import save_variable, load_variavle, data_generator_forward, data_generator_backward, Evaluator
from model import my_HAN_model

MAX_SENT_LENGTH = 50
MAX_SENTS = 350
EMBEDDING_DIM = 100

resample_flag = True
data = pd.read_csv("./data/Train.csv")
train = data[['SessionId','HighRiskFlag']].drop_duplicates().reset_index(drop=True)


# sub resample
if resample_flag == True:
    train_y1 = train[train['HighRiskFlag']==1]
    train_y0 = train[train['HighRiskFlag']==0]

    train_y0.reset_index(inplace=True)
    train_ = train_y0.sample(n=15000,random_state=123,axis=0)
    train = pd.concat([train_,train_y1])
    train = train.sample(frac=1, random_state=123).reset_index(drop=True)
    train.drop(['index'], axis=1, inplace=True)

def get_text(row):
    return list(row['Role']+":"+row['Text'])
tmp = data.groupby('SessionId').apply(get_text).rename('text')
train = pd.merge(train, tmp, on=['SessionId'], how='left', copy=False)

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

# 5折
train_batch_size = 16
batch_size = 32
kold = StratifiedKFold(random_state=2020,shuffle=True,n_splits=5).split(train, train['HighRiskFlag'])
for fold, (train_idx, valid_idx) in enumerate(kold):

    K.clear_session()
    md = my_HAN_model(MAX_SENT_LENGTH = MAX_SENT_LENGTH, MAX_SENTS=MAX_SENTS, embedding_matrix=embedding_matrix,vocab=vocab)
    model = md.create_model()

    train_generator = data_generator_forward(train.values[train_idx], train_batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS) 
    valid_generator = data_generator_forward(train.values[valid_idx], train_batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS)
    print("开始训练")

    evaluator = Evaluator(model_name='./model/best_model{}.weights'.format(fold), valid_generator=valid_generator, model=model)
    earlystop = EarlyStopping(monitor='val_recall',patience=5, mode='max')
    lr_reduce = ReduceLROnPlateau(monitor='val_recall', patience=2, verbose=1, factor=0.5, min_lr=1e-5, mode='max')
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[evaluator, lr_reduce, earlystop]
    )

kold = StratifiedKFold(random_state=2020,shuffle=True,n_splits=5).split(train, train['HighRiskFlag'])
for fold, (train_idx, valid_idx) in enumerate(kold):

    K.clear_session()
    md = my_HAN_model(MAX_SENT_LENGTH = MAX_SENT_LENGTH, MAX_SENTS=MAX_SENTS, embedding_matrix=embedding_matrix,vocab=vocab)
    model = md.create_model()

    train_generator = data_generator_backward(train.values[train_idx], train_batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS) 
    valid_generator = data_generator_backward(train.values[valid_idx], train_batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS)
    print("开始训练")

    evaluator = Evaluator(model_name='./model/best_han_model_rump_{}.weights'.format(fold), valid_generator=valid_generator, model=model)
    earlystop = EarlyStopping(monitor='val_recall',patience=5, mode='max')
    lr_reduce = ReduceLROnPlateau(monitor='val_recall', patience=2, verbose=1, factor=0.5, min_lr=1e-5, mode='max')
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[evaluator, lr_reduce, earlystop]
    )

