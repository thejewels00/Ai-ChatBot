This is a simple chatbot built using a machine learning model and Natural Language Processing (NLP) techniques. The chatbot is designed to understand user input, classify the intent, and respond with an appropriate message.

## Requirements

Before running the chatbot, make sure you have the following libraries installed:

- `tensorflow`
- `numpy`
- `nltk`
- `json`
- `pickle`

To install the necessary packages, you can use the following command:

```bash
pip install tensorflow numpy nltk
```

You will also need to download the necessary NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Files Needed

- `intents.json`: A JSON file that contains a list of intents and their corresponding responses.
- `words.pkl`: A pickle file that stores the vocabulary list (words) extracted from the training data.
- `classes.pkl`: A pickle file that stores the class labels (tags) of the intents.
- `chatbot.h5`: The pre-trained Keras model for intent classification.

## How It Works

1. **Data Preprocessing:**
   - The `clean_up_sentence()` function tokenizes the user input and lemmatizes each word to reduce them to their base form.
   - The `bag_of_words()` function converts the processed sentence into a bag of words representation, which is used as input to the model.

2. **Intent Classification:**
   - The `predict_class()` function takes the processed user input, converts it into a bag of words, and predicts the class (intent) with the trained model. It returns the intent with the highest probability.

3. **Response Generation:**
   - The `get_response()` function looks up the predicted intent in the `intents.json` file and randomly selects a response from the corresponding list.

4. **User Interaction:**
   - The chatbot waits for user input, processes the input to predict the intent, and then responds with an appropriate message.

## How to Run

1. Ensure that you have all the required files (`intents.json`, `words.pkl`, `classes.pkl`, `chatbot.h5`).
2. Run the Python script:

```bash
python chatbot.py
```

3. Start chatting with the chatbot! Type a message and see how the bot responds.

---

Let me know if you need any changes or further explanations!
