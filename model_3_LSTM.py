import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from gensim.models import KeyedVectors

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

from keras.models import *
from keras.layers import *
from keras.preprocessing.text import *
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

class eval_true_label():
    def __init__(self,model,test_dat):
        self.model = model
        self.test_dat = test_dat
        self.true_label = pd.read_csv('test_labels.csv').set_index('id')

    def evaluation(self):
        index = np.where(self.true_label.toxic>-1)[0]
        pred = self.model.predict(self.test_dat,verbose=1)
        pred = pred[index]
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
all = pd.concat([train,test],axis=0)
comment_text = all.comment_text
target = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]

comment_text_CHAR = []
for each in comment_text.values:
    temp = []
    for char in each:
        char = [x if x not in '!"#$%&\()*+, -./:;<=>?@[\\]^_`{|}~' else '' for x in char]
        temp.append(char[0].lower())
    comment_text_CHAR.append(" ".join([x for x in temp]))


tk = Tokenizer()
tk.fit_on_texts(comment_text.values)

comment_text_seq = tk.texts_to_sequences(comment_text.values)
comment_text_seq = pad_sequences(comment_text_seq,maxlen=400)
train_seq = comment_text_seq[np.where(all.toxic.notnull())]
test_seq = comment_text_seq[np.where(all.toxic.isnull())]


####################
inputs = Input(shape=(400,))
embedding = Embedding(input_dim=10000, output_dim=300)(inputs)
embedding = SpatialDropout1D(0.2)(embedding)
model = GRU(40,return_sequences = False, activation='relu', dropout=0.1,recurrent_dropout=0.1)(embedding)
# model = Bidirectional(GRU(128,return_sequences = True, activation='relu', dropout=0.1,recurrent_dropout=0.1))(embedding)
# model = Conv1D(80, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(model)
# temp_flat = Flatten()(model)
# temp_avg = GlobalAvgPool1D()(model)
# temp_max = GlobalMaxPool1D()(model)
# concat = concatenate([temp_avg,temp_max])

temp_max = Dense(40,activation='relu')(model)
temp_max = BatchNormalization()(temp_max)
###
model = Dense(6,activation='sigmoid')(temp_max)

model = Model(inputs=[inputs],outputs=[model])
model.summary()

model.compile(loss='binary_crossentropy',optimizer='Adam')

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(train_seq,target):
    model.fit(train_seq[train_index],target.values[train_index],epochs=6,batch_size=1024,\
               validation_data=(train_seq[test_index],target.values[test_index]))
