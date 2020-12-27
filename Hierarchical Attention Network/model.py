from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from bert4keras.backend import keras, set_gelu
from keras.layers import *
from keras import backend as K
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__( **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                shape=(input_shape[1], input_shape[1]),
                                initializer='uniform',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

import keras.backend as K

class SetLearningRate:
    """
    	layer learning rate
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # learning rate
        self.is_ada = is_ada # if adam

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embed', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)


# best GRU 100, no drop, 50word 350 sentence
class my_HAN_model(object):
    def __init__(self, MAX_SENT_LENGTH, MAX_SENTS, embedding_matrix, vocab):
    	self.MAX_SENT_LENGTH = MAX_SENT_LENGTH
    	self.MAX_SENTS = MAX_SENTS
    	self.embedding_matrix = embedding_matrix
    	self.vocab = vocab
    def create_model(self):
        sentence_inputs = Input(shape=(self.MAX_SENT_LENGTH,), dtype='float64')
        
        embed = Embedding(len(self.vocab) + 1, 100, input_length=self.MAX_SENT_LENGTH, weights=[self.embedding_matrix], trainable=True)
        embed = SetLearningRate(embed, 0.1, True)(sentence_inputs)
        l_lstm = Bidirectional(GRU(100, return_sequences=True))(embed)
        l_dense = TimeDistributed(Dense(200))(l_lstm)
        print(l_dense.shape)
        l_att = AttentionLayer()(l_dense)
        print(l_att.shape)
        sentEncoder = Model(sentence_inputs, l_att)
        
        review_input = Input(shape=(self.MAX_SENTS, self.MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
        l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
        l_att_sent = AttentionLayer()(l_dense_sent)
        outputs = Dense(1, activation='sigmoid')(l_att_sent)
        
        self.model = Model(review_input, outputs=outputs)
        self.model.summary()
        self.compile()
        return self.model
    
    def compile(self):
        self.model.compile(
             loss='binary_crossentropy',
            optimizer=Adam(1e-3),  
            metrics=['accuracy'],
        )

epsilon = 1e-5
smooth = 1
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)