import streamlit as st
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='chat_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Load the tokenizer JSON as a string
with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()

# Convert the JSON string back to a tokenizer
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Determine max_len from tokenizer data
max_len = max([len(x.split()) for x in tokenizer.word_index])

# Load label index
with open('label_index.json', 'r') as f:
    label_index = json.load(f)
labels = list(label_index.keys())

# Load responses from a separate JSON file or define them directly here
with open('responses.json', 'r') as f:
    responses = json.load(f)

print("Input shape expected by the model:", input_details[0]['shape'])

def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=18, padding='post')  # Ensure maxlen matches expected length
    padded = padded.astype('float32')  # Cast to float32
    if padded.shape[1] != 18:  # Check if the second dimension is correct
        raise ValueError(f"Expected sequence length 18, but got {padded.shape[1]}")
    if padded.ndim == 1:
        padded = np.expand_dims(padded, axis=0)
    elif padded.shape[0] != 1:
        padded = np.reshape(padded, (1, 18))  # Explicitly reshape to [1, 18]
    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data[0])
    tag = list(responses.keys())[predicted_index]
    return tag




# Streamlit app
# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app interface
st.title("Chatbot with History")

# Text input for user message
user_input = st.text_input("Type your message here:", "")

# Button to send the message
if st.button("Send"):
    if user_input:
        # Append user input to chat history
        st.session_state.chat_history.append(f"You: {user_input}")
        
        # Get bot response
        intent = predict_intent(user_input)
        response = np.random.choice(responses[intent])
        
        # Append bot response to chat history
        st.session_state.chat_history.append(f"Bot: {response}")
        
    
    # Clear input box after sending
    st.experimental_rerun()

# Display chat history
st.write("Chat History:")
for message in st.session_state.chat_history:
    st.text(message)