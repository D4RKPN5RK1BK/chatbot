# Копипаста из этого видео
# https://www.youtube.com/watch?v=1lwddP0KUEg

import json
import os
import pickle
import random
import nltk
import numpy as np
from pymorphy3 import MorphAnalyzer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

nltk.download('wordnet')
nltk.download('stopwords')

tokenizer = nltk.data.load("nltk:tokenizers/punkt/russian.pickle")
morph = MorphAnalyzer()

# использовать __file__ нормальное решние роблемы)
intentsPath = os.path.join(os.path.dirname(__file__), "intents.json")  # находим json файл с нужными данными
intents = json.loads(open(intentsPath, encoding="utf-8").read())  # вытягиваем данные из файла

print("path:", intentsPath)

all_words = []  # Массив всех слов в intent.json
all_tags = []  # Массив всех тэгов в intent.json
documents = []  # Тэг + привязанные к нему слова
ignore_letters = [".", ","]  # Пока для простоты опустим

for intent in intents:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        all_words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in all_tags:
            all_tags.append(intent["tag"])

normalized_words = [morph.normal_forms(word)[0] for word in all_words]
normalized_words = sorted(set(normalized_words))

pickle.dump(normalized_words, open("/Temp/words.pkl", "wb"))
pickle.dump(all_tags, open("/Temp/tags.pkl", "wb"))

# Создаем массив из нулей, потом ищем по названию тэга, берем индекс и вместо нуля по индексу вставляем еденицу
tags_match_template = [0] * len(all_tags)
training_data = []

for document in documents:
    words_for_tag = []
    word_patterns = document[0]
    word_patterns = [morph.normal_forms(word)[0] for word in word_patterns]
    for word in normalized_words:
        words_for_tag.append(1 if word in word_patterns else 0)

    output_row = list(tags_match_template)
    output_row[all_tags.index(document[1])] = 1
    training_data.append([words_for_tag, output_row])

random.shuffle(training_data)  # Тоссуем модели в случайном порядке

# todo создание моделеи на основе туплов (массивов туплов) устарело, так что тут нужно посмотреть какой тип и
#   привести все к красоте.
#   А это вообще тут нужно? Вроде как обчно используется для создания массива, а тут для преобразования?
training_data = np.array(training_data, dtype=object)

train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])

# todo все что ниже идет это иероглифы, потом расписать все подробнов коментах

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('/Temp/chatbot_model.h5', hist)
print("Model creation done.")
