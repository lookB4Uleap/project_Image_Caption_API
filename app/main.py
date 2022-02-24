import os, shutil
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from tensorflow.keras.models import Sequential, Model
# from keras.utils import np_utils
from tensorflow.keras.preprocessing import image, sequence
import cv2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


vocab = np.load('./vocab.npy', allow_pickle=True)
vocab = vocab.item()
inv_vocab = {v:k for k,v in vocab.items()}

embedding_size = 128
vocab_size = len(vocab)
max_len = 35

# image_model = Sequential()

# image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
# image_model.add(RepeatVector(max_len))


# language_model = Sequential()

# language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
# language_model.add(LSTM(256, return_sequences=True))
# language_model.add(TimeDistributed(Dense(embedding_size)))


# conca = Concatenate()([image_model.output, language_model.output])
# x = LSTM(128, return_sequences=True)(conca)
# x = LSTM(512, return_sequences=False)(x)
# x = Dense(vocab_size)(x)
# out = Activation('softmax')(x)
# model = Model(inputs=[image_model.input, language_model.input], outputs = out)

# model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model = load_model('./image_caption_generator_model.h5')
model.load_weights('./model_weights.h5')

resnet = load_model('./resnet50.h5')

app = Flask(__name__, static_url_path = "/app/static", static_folder = "static")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
CORS(app)

folder = './app/static/'

@app.route("/")
def hello_world():
    return "<h1>Hello, World! " + str(vocab_size) + "</h1>"

@app.route('/api/generate-caption', methods = ['GET', 'POST'])
def generate_caption():
  if request.method == 'POST' :
    f = request.files['image']
    for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
      except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
    f.save('./app/static/' + f.filename)
    # print(f)

    global model, resnet, vocab, inv_vocab, max_len, embedding_size
    image = cv2.imread('./app/static/' + str(f.filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    image = np.reshape(image, (1,224,224,3))

    incept = resnet.predict(image).reshape(1,2048)

    text_inp = ['startofseq']

    count = 0
    caption = ''
    while tqdm(count < 25):
        count += 1
        encoded = []
        for i in text_inp:
            encoded.append(vocab[i])
        # encoded = [encoded]
        # encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)
        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)
        prediction = np.argmax(model.predict([incept, padded]))
        sampled_word = inv_vocab[prediction]
            
        if sampled_word != 'endofseq':
            caption = caption + ' ' + sampled_word

        text_inp.append(sampled_word)

    return jsonify({
      'name'  : f.filename,
      'caption' : caption
    })