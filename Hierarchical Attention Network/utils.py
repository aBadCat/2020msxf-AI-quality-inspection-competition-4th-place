import _pickle as pickle
import numpy as np
import jieba
import tensorflow as tf
import keras
from sklearn import metrics

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename
 
def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def sequence_padding2d(inputs, length1, length2, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0* length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)

# first 350 sentences
def word2idx_forward(doc, vocab, MAX_SENT_LENGTH, MAX_SENTS):
    sents = []
    for sent in doc:
        sent = [vocab[word] for word in jieba.lcut(sent) if word in vocab]
        sents.append(sent[:MAX_SENT_LENGTH] + [0] * (MAX_SENT_LENGTH - len(sent)))
    sents = np.array(sents[:MAX_SENTS])
    if len(sents) == MAX_SENTS:
        return sents
    else:
        return np.r_[sents, np.array([[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(sents)))]

# last 350 sentences
def word2idx_backward(doc, vocab, MAX_SENT_LENGTH, MAX_SENTS):
    sents = []
    for sent in doc[::-1]:
        sent = [vocab[word] for word in jieba.lcut(sent) if word in vocab]
        sents.append(sent[:MAX_SENT_LENGTH] + [0] * (MAX_SENT_LENGTH - len(sent)))
    sents = np.array(sents[:MAX_SENTS])
    if len(sents) == MAX_SENTS:
        return sents[::-1]
    else:
        return np.r_[sents[::-1], np.array([[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(sents)))]

def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


class data_generator_forward(object):
    """数据生成器
    """
    def __init__(self, data, batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS, train = True,buffer_size = None):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.vocab = vocab
        self.MAX_SENT_LENGTH = MAX_SENT_LENGTH
        self.MAX_SENTS = MAX_SENTS
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000
    def __len__(self):
        return self.steps
    
    def sample(self, random=False):
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield chaches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                        while caches:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]
            data = generator()
        else:
            data = iter(self.data)
        
        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next
        
        yield True, d_current
        
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if self.train == True:
            for is_end, (label, ids, text) in self.sample(random):
                batch_token_ids.append(word2idx_forward(text, self.vocab, self.MAX_SENT_LENGTH, self.MAX_SENTS))
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield batch_token_ids, batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        else:
            for is_end, (ids, text, label) in self.sample(random):
                batch_token_ids.append(word2idx_forward(text, self.vocab, self.MAX_SENT_LENGTH, self.MAX_SENTS))
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    yield batch_token_ids
                    batch_token_ids = []
        
    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d
    
    def forpredict(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d

class data_generator_backward(object):
    """数据生成器
    """
    def __init__(self, data, batch_size,vocab, MAX_SENT_LENGTH, MAX_SENTS, train = True,buffer_size = None):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.vocab = vocab
        self.MAX_SENT_LENGTH = MAX_SENT_LENGTH
        self.MAX_SENTS = MAX_SENTS
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000
    def __len__(self):
        return self.steps
    
    def sample(self, random=False):
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield chaches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                        while caches:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]
            data = generator()
        else:
            data = iter(self.data)
        
        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next
        
        yield True, d_current
        
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if self.train == True:
            for is_end, (label, ids, text) in self.sample(random):
                batch_token_ids.append(word2idx_backward(text, self.vocab, self.MAX_SENT_LENGTH, self.MAX_SENTS))
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield batch_token_ids, batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        else:
            for is_end, (ids, text, label) in self.sample(random):
                batch_token_ids.append(word2idx_backward(text, self.vocab, self.MAX_SENT_LENGTH, self.MAX_SENTS))
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    yield batch_token_ids
                    batch_token_ids = []
        
    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d
    
    def forpredict(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d

def evaluate(data,model):
    total, right = 0., 0.
    y_trues = []
    y_preds = []
    for x_true, y_true in data:
        y_preds = np.r_[y_preds, model.predict(x_true)[:,0]]
        y_trues = np.r_[y_trues, y_true[:, 0]]
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
    return  metrics.auc(fpr, tpr)

def recallTopK(y_trues, y_preds, K=1000):
    y_preds = y_preds.squeeze()
    y_trues = y_trues.squeeze()
    y_preds = (-y_preds).argsort()[:K]
    y_trues_1 = [x for x in range(len(y_trues)) if y_trues[x] == 1]
    tmp = [val for val in y_preds if val in y_trues_1]
    if len(y_trues_1) > K:
        return len(tmp) / K
    else:
        return len(tmp) / len(y_trues_1)

def evaluate_recall(data,model):
    total, right = 0., 0.
    y_trues = []
    y_preds = []
    for x_true, y_true in data:
        y_preds = np.r_[y_preds, model.predict(x_true)[:,0]]
        y_trues = np.r_[y_trues, y_true[:, 0]]
#     fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
    return  recallTopK(y_trues, y_preds)

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_name,valid_generator,model):
        self.best_val_acc = 0
        self.model_name = model_name
        self.valid_generator = valid_generator
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.valid_generator, self.model)
        val_recall = evaluate_recall(self.valid_generator, self.model)
        if (val_recall > self.best_val_acc):
            print('1')
            self.best_val_acc = val_recall
            self.model.save_weights(self.model_name)
        print(
            u'val_auc: %.5f, best_val_auc: %.5f, val_recall: %.5f' %
            (val_acc, self.best_val_acc, val_recall)
        )
        logs['val_auc'] = val_acc
        logs['val_recall'] = val_recall

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed