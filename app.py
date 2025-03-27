import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word.h5")

with open("tokenizer.pickle","rb") as file:
    tokenizer = pickle.load(file)

def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding="pre")

    if token_list is None or token_list.size == 0:
        print("Error: Token list is empty or improperly formed")
        return None
    
    predicted = model.predict(token_list,verbose=0)# Ensure model is returning valid output
    if predicted is None:
         print("Error: Model did not return a valid prediction")
         return None

    predicted_word_index = np.argmax(predicted,axis=1)[0] # Ensure correct index retrieval

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    print("Error: No word found for predicted index")
    return None


 #streamlit
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")

if st.button("predict next word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')