import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from gensim.models import KeyedVectors

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_preprocessing = pd.read_pickle('train_preprocessing.pkl')
test_preprocessing = pd.read_pickle('test_preprocessing.pkl')

train.head()

from keras.models import *
from keras.layers import *
from keras.preprocessing.text import *
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

class eval_true_label():
    def __init__(self,model,test_dat,index):
        self.model = model
        self.test_dat = test_dat
        self.true_label = pd.read_csv('test_labels.csv').set_index('id')
        self.index = index

    def evaluation(self):
        pred = self.model.predict(self.test_dat,verbose=1)
        pred = pred[self.index]
        self.true_label = self.true_label.loc[self.true_label.toxic>-1]
        res = []
        for i in range(6):
            res.append(roc_auc_score(self.true_label.values[:,i],pred[:,i]))
        print('Submission Score is %s' %str(np.mean(res)))

class columns_auc(Callback):
    def __init__(self,train_X_dat,train_y_dat,val_X_dat,val_y_dat):
        self.train_x = train_X_dat
        self.train_y = train_y_dat
        self.val_x = val_X_dat
        self.val_y = val_y_dat

    def on_epoch_end(self,epoch,logs={}):
        train_y_pred = self.model.predict(self.train_x)
        val_y_pred = self.model.predict(self.val_x)

        res_tr = []
        res_te = []
        for i in range(6):
            res_tr.append(roc_auc_score(self.train_y.values[:,i],train_y_pred[:,i]))
            res_te.append(roc_auc_score(self.val_y.values[:,i],val_y_pred[:,i]))

        res_tr = np.mean(res_tr)
        res_te = np.mean(res_te)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(res_tr,4)),str(round(res_te,4))),end=100*' '+'\n')

######################################################
embeddings_index_glove = {}
with open('glove.840B.300d.txt',encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index_glove[word] = coefs

            #########################
embeddings_index_fasttext = {}
with open('wiki-news-300d-1M.vec',encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index_fasttext[word] = coefs

            #########################
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

del word2vec
######################################################
all = pd.concat([train,test],axis=0)
comment_text = all.comment_text
target = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]

tk = Tokenizer()
tk.fit_on_texts(comment_text.values)

comment_text_seq = tk.texts_to_sequences(comment_text.values)
comment_text_seq = pad_sequences(comment_text_seq,maxlen=200)
train_seq = comment_text_seq[np.where(all.toxic.notnull())]
test_seq = comment_text_seq[np.where(all.toxic.isnull())]

#
embedding_matrix_glove = np.zeros((30000, 300))
for word, i in tk.word_index.items():
    if i >= 30000:
        continue
    embedding_vector = embeddings_index_glove.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_glove[i] = embedding_vector
#
embedding_matrix_fasttext = np.zeros((30000, 300))
for word, i in tk.word_index.items():
    if i >= 30000:
        continue
    embedding_vector = embeddings_index_fasttext.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_fasttext[i] = embedding_vector
# #
# embedding_matrix_w2v = np.zeros((30000, 300))
# for word, i in tk.word_index.items():
#     if i >= 30000:
#         continue
#     embedding_vector = embedding_matrix_w2v.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix_w2v[i] = embedding_vector

### Training via Word Embedding
inputs = Input(shape=(200,))

embedding_glove = Embedding(input_dim=30000, output_dim=300, weights=[embedding_matrix_glove],trainable = False)(inputs)
embedding_glove = SpatialDropout1D(0.3)(embedding_glove)
model = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(embedding_glove)
temp_max_glove = GlobalMaxPool1D()(model)

embedding_fasttext = Embedding(input_dim=30000, output_dim=300, weights=[embedding_matrix_fasttext],trainable = False)(inputs)
embedding_fasttext = SpatialDropout1D(0.2)(embedding_fasttext)
model = Conv1D(78, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform")(embedding_fasttext)
temp_max_fasttext = GlobalMaxPool1D()(model)

# embedding_w2v = Embedding(input_dim=30000, output_dim=300, weights=[embedding_matrix_w2v],trainable = False)(inputs)
# embedding_w2v = SpatialDropout1D(0.4)(embedding_w2v)
# model = Conv1D(80, kernel_size = 4, padding = "valid", kernel_initializer = "glorot_uniform")(embedding_w2v)
# temp_max_w2v = GlobalMaxPool1D()(model)

concat_emb = concatenate([temp_max_glove,temp_max_fasttext])

### Training via Comments Statistics
inputs2 = Input(shape=[train_preprocessing.shape[1]])
dense = Dense(54,activation='relu')(inputs2)
dense = BatchNormalization()(dense)
dense = Dense(23,activation='relu')(inputs2)
dense = BatchNormalization()(dense)

###
concat = Concatenate(axis=1)([concat_emb,dense])
concat = Dense(40,activation='relu')(concat)
concat = BatchNormalization()(concat)
###
model = Dense(6,activation='sigmoid')(concat)

model = Model(inputs=[inputs, inputs2],outputs=[model])
model.summary()

model.compile(loss='binary_crossentropy',optimizer='Adam')


#############

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(train_seq,target):

    model.fit([train_seq[train_index],train_preprocessing.values[train_index]],target.values[train_index],epochs=8,batch_size=1024,\
               validation_data=([train_seq[test_index],train_preprocessing.values[test_index]],target.values[test_index]))
