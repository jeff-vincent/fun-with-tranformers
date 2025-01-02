import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Positional encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.get_positional_encoding(sequence_length, embed_dim)

    def get_positional_encoding(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]
        return self.embedding(x) + self.pos_encoding[:, :length, :]

# Transformer-based text generation model
class TextGenerator(Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, sequence_length):
        super(TextGenerator, self).__init__()
        self.positional_encoding = PositionalEncoding(sequence_length, vocab_size, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dense = Dense(vocab_size)

    def call(self, x, training):
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return self.dense(x)

# Hyperparameters
vocab_size = 1000  # Set based on your dataset
embed_dim = 64
num_heads = 4
ff_dim = 128
num_layers = 2
sequence_length = 50

# Create the model
model = TextGenerator(vocab_size, embed_dim, num_heads, ff_dim, num_layers, sequence_length)
model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Sample training data
text = "hello world welcome to text generation with transformers"
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts([text])

# Tokenize and prepare input sequences
sequences = tokenizer.texts_to_sequences([text])[0]
input_sequences = []
for i in range(1, len(sequences)):
    n_gram_sequence = sequences[:i + 1]
    input_sequences.append(n_gram_sequence)

input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=sequence_length, padding='pre')
X = input_sequences[:, :-1] 
y = input_sequences[:, 1:]


# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Text generation function
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predictions = model.predict(token_list)
        predicted_word_index = tf.argmax(predictions[0, -1, :]).numpy()
        output_word = tokenizer.index_word.get(predicted_word_index, '')
        seed_text += " " + output_word
    return seed_text

# Generate new text
seed_text = "hello world"
next_words = 10
print(generate_text(seed_text, next_words, sequence_length))
