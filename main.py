import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

# training the model
import tflearn
import tensorflow as tf
# generate random responses for the user input
import random
import json
# pickle is used to save any model or intermediate structures
import pickle

from flask import Flask, render_template, request

#nltk.download('punkt')
#nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
        words = []
        labels = []
        docs_x = []
        docs_y = []
        ignore_letters = [',', '.', '?', '!']
# loop through each sentence in the intents patterns
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                # tokenize each and every word in the sentence
                wrds = nltk.word_tokenize(pattern)
                # add word to the words list
                words.extend(wrds)
                # add word(s) to the document
                docs_x.append(wrds)

                docs_y.append(intent["tag"])
            # add tags to the labels list
            if intent["tag"] not in labels:
                    labels.append(intent["tag"])
        # lemmatize and convert each word into lower case and remove duplicates
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
        words = sorted(list(words))
        # remove duplicate labels
        labels = sorted(labels)

        # create training data
        training = []
        output = []
        # creating an empty array
        out_empty = [0 for _ in range(len(labels))]
        # create training set, bag of words for each sentence
        for x, doc in enumerate(docs_x):
            # initializing bag of words
            bag = []
            # lemmatizing each word
            wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]
            # create bag of words array
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
            # current tag as 1 and the rest as 0s
            # generate ouput
            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = np.array(training)
        output = np.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)
# reseting underlying graph data
tf.compat.v1.reset_default_graph()

# Building neural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Define model and set up tensorboard
model = tflearn.DNN(net)

# if model is already trained and save
try:
   model.load("model.tflearn")
except:
    # start training
    model.fit(training, output, n_epoch=5000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    # cleaning user input sentence
    # tokenizing the pattern
    s_words = nltk.word_tokenize(s)
    # lemmatizing each word
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    # generating bag of words
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)
def predict_class(s, model):
    # filter predictions below a threshold
    p = bag_of_words(s, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.3
    results = [[i, r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    #sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag']== tag) :
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, data)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()