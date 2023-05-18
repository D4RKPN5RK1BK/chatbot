import json
import os
import pickle
import random

import nltk
import numpy as np
from pymorphy3 import MorphAnalyzer
from tensorflow.python.keras.models import load_model

morph = MorphAnalyzer()
intentsPath = os.path.join(os.path.dirname(__file__), "intents.json")
intents = json.loads(open(intentsPath, encoding="utf-8").read())

words = pickle.load(open("/Temp/words.pkl", "rb"))
tags = pickle.load(open("/Temp/tags.pkl", "rb"))

model = load_model("/Temp/chatbot_model.h5")


def cleanup_sentence(sentence=''):
    sentence_words = nltk.word_tokenize(sentence)
    lemmatized_words = [morph.normal_forms(word)[0] for word in sentence_words]
    return lemmatized_words


def bag_of_words(sentence=''):
    sentence_words = cleanup_sentence(sentence)
    bag = [0] * len(words)

    for word in sentence_words:
        for i, w in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict(sentence=''):
    bag = bag_of_words(sentence)
    prediction = model.predict(np.array([bag]))[0]
    ERROR_TRESHHOLD = 0.25
    result = [[i, r] for i, r in enumerate(prediction) if r > ERROR_TRESHHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    result_list = [{"intent": tags[r[0]], "prediction": str(r[1])} for r in result]
    return result_list


def get_responce_by_tag(tag):
    responses = next(intent['responses'] for intent in intents if intent['tag'] == tag)
    tag_response = random.choice(responses)
    return tag_response


def response_to_message(message):
    predicted_sentence = predict(message)

    if any(predicted_sentence):
        predicted_tag = predicted_sentence[0]['intent']
        return get_responce_by_tag(predicted_tag)

    return get_responce_by_tag('unknown')


while True:
    message = input()
    if input == "q":
        break
    response = response_to_message(message)
    print(response)

