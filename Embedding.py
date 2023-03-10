import numpy as np
import tensorflow as tf
from tensorflow import distribute

def get_text_embeddings(text_list, vocab_size=150, output_dim=100):
    # Initialize the tokenizer
    text_list= text_list.astype(str)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
  
    # Fit the tokenizer on the text corpus
    tokenizer.fit_on_texts(text_list)

    # Convert the text corpus to sequences
    sequences = tokenizer.texts_to_sequences(text_list)

    # Pad the sequences to the same length
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)

    # Define the embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=output_dim)

    st = distribute.MirroredStrategy()
    with st.scope():
        # Create a sequential model
        model = tf.keras.Sequential()

        # Add the embedding layer as the first layer
        model.add(embedding_layer)

        # Get the embeddings
        embeddings = model.predict(padded_sequences)
    return embeddings