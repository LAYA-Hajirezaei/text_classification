import pandas as pd
import re
IN_PATH = "all-chat-clean.xlsx"

df = pd.ExcelFile(IN_PATH)
df1 = df.parse("Sheet1",encoding="utf-8")
print(df1)


chatId = []
values = []
for rownum in range(len(df1)):
    chatid = df1.iat[rownum,0]
    row = df1.iat[rownum, 1]
    for x in row.split('|'):
        if re.match(r"[^|0-9|]\D+", x):
            chatId.append(chatid)
            values.append(x)
data = pd.DataFrame()
data['chatid'] = chatId
data['value'] = values
print(data)

data=data[~data['value'].str.contains('support')]
data=data[~data['value'].str.contains('salimi')]
data=data[~data['value'].str.contains('ایرانژاد')]
data=data[~data['value'].str.contains(' سهند مددی')]
data=data[~data['value'].str.contains('صالح')]
data=data[~data['value'].str.contains('رضایی')]
data=data[~data['value'].str.contains('ابوطالبی')]
data=data[~data['value'].str.contains('محمدی')]
data=data[~data['value'].str.contains('شعبانی')]
data=data[~data['value'].str.contains('شایسته')]
data=data[~data['value'].str.contains('کبیری')]
data=data[~data['value'].str.contains('مقدم')]
data=data[~data['value'].str.contains(' دست ورز ')]
data=data[~data['value'].str.contains('محمدزاده')]

new_data=data.groupby('chatid').agg({'value': lambda x:' ,'.join(x)})
print(new_data)

new_data.to_excel('test_for_LSTM.xlsx')

import xlrd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
# from stopwords_guilannlp import stopwords_output

train_data=xlrd.open_workbook('train_for_remove_support.xlsx')
train_data=pd.read_excel(train_data)
# print(train_data.head())

print(train_data.groupby('label').describe())

category_y = train_data['label']
print(category_y)

category_le = LabelEncoder()
train_data['label'] = category_le.fit_transform(train_data['label'])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train_data['label'] = le.fit_transform(train_data['label'])

y_train = train_data['label']
print(y_train)

test_data=xlrd.open_workbook('test_for_LSTM.xlsx')
test_data=pd.read_excel(test_data)
# print(test_data.head())


cat_vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,1), min_df=0.005)
# train set
x_train_corpus = train_data['value'].values
# x_train_corpus = preprocess(x_train_corpus)
# test set
x_test_corpus = test_data['value'].values
# x_test_corpus = preprocess(x_test_corpus)

# Fit vectorizer
cat_vectorizer.fit(x_train_corpus)

# Print vocabulary
cat_vocab = cat_vectorizer.get_feature_names()
print(cat_vocab)
print('Vocab length:', len(cat_vocab))


x_train_vec = cat_vectorizer.transform(x_train_corpus).toarray()
x_test_vec = cat_vectorizer.transform(x_test_corpus).toarray()

print('Shape of x_train vector for category predictor: ', x_train_vec.shape)
print('Shape of x_test vector for category predictor: ', x_test_vec.shape)



import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Convolution1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.applications import VGG16
import numpy as np
from keras.metrics import categorical_accuracy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


lstm_x_train = train_data['value'].values
# lstm_x_train = preprocess(lstm_x_train)
# test set
lstm_x_test = test_data['value'].values
# lstm_x_test = preprocess(lstm_x_test)

num_words = 10000

# create the tokenizer
tokenizer = Tokenizer(num_words=num_words)

# fit the tokenizer on the documents
tokenizer.fit_on_texts(x_train_corpus)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(lstm_x_train)

# pad sequences
max_length = max([len(s.split()) for s in lstm_x_train])
x_train_padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

encoded_docs = tokenizer.texts_to_sequences(lstm_x_test)
x_test_padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

y_train_classes = np.unique(y_train)
y_train_classes_len = len(y_train_classes)

y_test_classes = np.unique(y_train)
y_test_classes_len = len(y_test_classes)

from keras.utils import to_categorical
#
categorical_y_train = to_categorical(y_train, y_train_classes_len)


model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Bidirectional(LSTM(50, return_sequences=True, name='lstm_layer')))
model.add(GlobalMaxPool1D())
model.add(Dropout(rate=0.5))
model.add(Dense(100, activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(y_train_classes_len, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[categorical_accuracy])
model.summary()


batch_size =2
epochs = 2
hist = model.fit(x_train_padded, categorical_y_train, batch_size=batch_size, epochs=epochs)

y_pred=model.predict_classes(x_test_padded)
print(y_pred)

prediction=list(category_le.inverse_transform(y_pred))
print(prediction)

test_data['prediction']=prediction
print(test_data)

test_data['prediction'].value_counts()

# # svaing the model
# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# # loading the model
# from keras.models import load_model
# model = load_model('my_model.h5')

