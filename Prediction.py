import pandas as pd

import pickle




def prediction(text):
    text = pd.Series(text)

    text = pd.Series(text)
    vectoriser = pickle.load(open("vectorizer.pkl", "rb"))
    text = vectoriser.transform(text)
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict(text)
    prediction = prediction[0]

    sentiments = {1:"religion",2: "age",3: "ethnicity",4: "gender",
                        5 : "other_cyberbullying",6: "not_cyberbullying"}

    for i in sentiments.keys():
        if i == prediction:
            return sentiments[i]
