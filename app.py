import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_predection_LSTM.h5')

with open('tokenizer.h5', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_the_next_word(seed_text, tokenizer, next_words=1, max_sequence_len=14):
  for _ in range(next_words):
    # max_sequence_len =  max(len(x) for x in seed_text.split())
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if len(token_list) >= max_sequence_len:
      token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]


    output_word = None
    for word, index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
        break
    if output_word is None:
        break

    seed_text += " " + output_word

  return seed_text



# st.title("Next Word Prediction using LSTM and Keras")
# input_text = st.text_input("Enter your text here")
# input_next_word = st.number_input("Enter number of words to be predicted", min_value=1, max_value=10, value=1, step=1)



# if st.button("Predict"):
#     if input_text.strip() == "":
#         st.write("Please enter some text to predict the next word.")
#     else:
#         predicted_text = predict_the_next_word(input_text, tokenizer, next_words=input_next_word)
#         st.write("Predicted text: ", predicted_text)



# Set page config
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="wide",
)

# Main header
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #4B0082;'>🤖 Next Word Prediction App</h1>
        <p style='font-size: 18px; color: #555;'>Powered by LSTM & Keras</p>
    </div>
    """, unsafe_allow_html=True
)

# Input section in two columns
col1, col2 = st.columns([3, 1])

with col1:
    input_text = st.text_area("Enter your seed text here:", placeholder="Type something to start...", height=120)

with col2:
    input_next_word = st.slider("Number of words to predict:", 1, 10, 1)

# Predict button
if st.button("Predict Next Words", use_container_width=True):
    if input_text.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        result = predict_the_next_word(input_text, tokenizer, input_next_word)
        # Display result in a nice card style
        st.markdown(
            f"""
            <div style='background-color: #f0f0f5; padding: 15px; border-radius: 10px;'>
                <h3 style='color: #222;'>Predicted Text:</h3>
                <p style='font-size: 20px; color: #4B0082;'>{result}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Optional: Show original input in an expander
with st.expander("Show Original Input"):
    st.info(input_text)

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px; color: #888;'>
        Made with ❤️ using Python, Streamlit & Keras
    </p>
    """,
    unsafe_allow_html=True
)
