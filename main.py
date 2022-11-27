import csv
import copyreg
import configparser
import pathlib
import json
import random
import numpy
import tflearn
import pickle
import tensorflow
from datetime import datetime

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')


with open("intents.json") as file:
    data = json.load(file)

words_list = []
labels_list = []
patient_attributes = {
    "name": "John Smith",
    "date_of_birth": "2021/01/01",
    "gender": "Male",
    "children": "Jack Smith, Jane Smith",
    "spouse": "wife: Rachel Smith",
    "residence": "Toronto, Ontario",
    "hobbies": "swimming, cooking",
    "hospital": "Toronto Western Hospital"
}
loved_one_attributes = {
    "name": "Jack Smith",
    "date_of_birth": "2021/01/02",
    "gender": "Male",
    "children": "",
    "spouse": "",
    "residence": "Toronto, Ontario",
    "hobbies": "writing, reading"
}

def train_model(data):
    # Add "x" below the try if modified intents.json to re-train the model
    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
        words_list.extend(words)
        labels_list.extend(labels)
    except:
        words = []
        labels = []
        docs_of_words = []
        docs_of_intents = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_of_words.append(wrds)
                docs_of_intents.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        output_empty = [0 for _ in range(len(labels))]

        for i, doc in enumerate(docs_of_words):
            group = []

            wrds = [stemmer.stem(w) for w in doc]

            for w in words:
                if w in wrds:
                    group.append(1)
                else:
                    group.append(0)

            out_row = output_empty[:]
            out_row[labels.index(docs_of_intents[i])] = 1

            training.append(group)
            output.append(out_row)

        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

        words_list.extend(words)
        labels_list.extend(labels)

    tensorflow.reset_default_graph()

    network = tflearn.input_data(shape=[None, len(training[0])])
    network = tflearn.fully_connected(network, 8)
    network = tflearn.fully_connected(network, 8)
    network = tflearn.fully_connected(network, len(output[0]), activation="softmax")
    network = tflearn.regression(network)

    model = tflearn.DNN(network)

    # Add "x" below the try if modified intents.json to re-train the model
    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")

    return (model, data["intents"])

def get_possible_responses():
    responses = ["I see", "Mhmm", "Oh okay", "Ya?"]

    for intent in data["intents"]:
        for response in intent["responses"]:
            possible_response = add_personalized_info(intent["tag"], response)
            responses.append(possible_response)

    return responses

def add_personalized_info(tag, response):
    if tag == "patientAge":
        year_of_birth = patient_attributes["date_of_birth"].split('/')[0]
        current_year = datetime.now().year
        age = current_year - int(year_of_birth)
        response += str(age)
    elif tag == "patientLocation":
        response += patient_attributes["hospital"] + " in " + patient_attributes["residence"]
    elif tag == "patientName":
        response += patient_attributes["name"]
    elif tag == "patientHobbies":
        if patient_attributes["hobbies"] == "":
            response = "You don't really have any hobbies"
        else:
            hobbies = patient_attributes["hobbies"].split(", ")
            for i in range(len(hobbies) - 1):
                response += hobbies[i] + ", "
            if len(hobbies) > 1:
                response += "and "
            response += hobbies[len(hobbies) - 1]
    elif tag == "patientChildren":
        if patient_attributes["children"] == "":
            response = "You don't have kids"
        else:
            children = patient_attributes["children"].split(", ")
            for i in range(len(children) - 1):
                response += children[i].split(" ")[0] + ", "
            if len(children) > 1:
                response += "and " + children[len(children) - 1].split(" ")[0] + " are "
            else:
                response += children[len(children) - 1].split(" ")[0] + " is "
            response += "doing great"
    elif tag == "patientSpouse":
        if patient_attributes["spouse"] == "":
            response = "You don't have a spouse"
        else:
            response += patient_attributes["spouse"].split(" ")[1] + " is doing great"
    elif tag == "lovedOneAge":
        year_of_birth = loved_one_attributes["date_of_birth"].split('/')[0]
        current_year = datetime.now().year
        age = current_year - int(year_of_birth)
        response += str(age)
    elif tag == "lovedOneLocation":
        response += loved_one_attributes["residence"]
    elif tag == "lovedOneName":
        response += loved_one_attributes["name"]
    elif tag == "lovedOneHobbies":
        if loved_one_attributes["hobbies"] == "":
            response = "I don't really have any hobbies"
        else:
            hobbies = loved_one_attributes["hobbies"].split(", ")
            for i in range(len(hobbies) - 1):
                response += hobbies[i] + ", "
            if len(hobbies) > 1:
                response += "and "
            response += hobbies[len(hobbies) - 1]
    elif tag == "lovedOneChildren":
        if loved_one_attributes["children"] == "":
            response = "I don't have kids"
        else:
            children = loved_one_attributes["children"].split(", ")
            for i in range(len(children) - 1):
                response += children[i].split(" ")[0] + ", "
            if len(children) > 1:
                response += "and " + children[len(children) - 1].split(" ")[0] + " are "
            else:
                response += children[len(children) - 1].split(" ")[0] + " is "
            response += "doing great"
    elif tag == "lovedOneSpouse":
        if loved_one_attributes["spouse"] == "":
            response = "I don't have a spouse"
        else:
            response += loved_one_attributes["spouse"].split(" ")[1] + " is doing great"
    elif tag == "time":
        response += str(datetime.now().hour) + " " + str(datetime.now().minute)
    return response

def generate_response(model, inp):
    results = model.predict([group_of_words(inp, words_list)])[0]
    results_index = numpy.argmax(results)
    tag = labels_list[results_index]

    if results[results_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        response = random.choice(responses)
        response = add_personalized_info(tag, response)
    else:
        responses = ["I see", "Mhmm", "Oh okay", "Ya?"]
        response = random.choice(responses)

    print(response)
    return response

def group_of_words(s, words):
    group = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                group[i] = 1

    return numpy.array(group)

trained_model = train_model(data)[0]
generate_response(trained_model, "what time is it")
