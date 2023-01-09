import tensorflow as tf
import horovod.tensorflow.keras as hvd
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

hvd.init()


df = pd.read_csv("./Airline-Tweets.csv",encoding='latin1')
nf = pd.read_csv("./NBAdata.csv",encoding='latin1')
review_df = df[['text','airline_sentiment']]
review_df = review_df[review_df['airline_sentiment'] != 'neutral']

review_nf = nf[['tweets','label']]

sentiment_label = review_df.airline_sentiment.factorize()


sentiment_label_first = sentiment_label[0]
sentiment_label_train = sentiment_label_first[:int(0.8*len(sentiment_label_first))]%hvd.size()
sentiment_label_test = sentiment_label_first[int(0.8*len(sentiment_label_first)):]%hvd.size()

tweet_train = review_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet_train)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet_train)

padded_sequence = pad_sequences(encoded_docs, maxlen=200)

padded_sequence_train = padded_sequence[0:int(0.8*len(padded_sequence))]%hvd.size()
padded_sequence_test = padded_sequence[int(0.8*len(padded_sequence)):]%hvd.size()


embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

scaled_lr = 0.001 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)
opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'],experimental_run_tf_function=False)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
]


# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

history = model.fit(padded_sequence_train,sentiment_label_train,steps_per_epoch=500 // hvd.size(),
validation_data=(padded_sequence_test,sentiment_label_test),callbacks = callbacks, epochs=10, batch_size=32,verbose = verbose)

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plt.jpg")

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print('text: '+ text + 'predicted label: ' + sentiment_label[1][prediction])
test_sentence1 = "Happy NFL draft eve. Here's a look at what to know about Trey McBride's projections and other former CSU players in the mix to join the NFL." #positive
predict_sentiment(test_sentence1)
test_sentence2 = "Until Kevin Durant has an opportunity to right this four-game wrong, I can no longer make the case he's The Best Player on the Planet." #Negative
predict_sentiment(test_sentence2)
test_sentence3 = "The Denver Nuggets have been eliminated from the playoffs." #neg
predict_sentiment(test_sentence3)
test_sentence4 = "Nothing but respect between Kevin Durant and Jayson Tatum." #positive
predict_sentiment(test_sentence4)
test_sentence5 = "Kevin Durant is one of the 5 greatest shooters I've seen. As a total player he can't impact games like Giannis, who can beat you with scoring, rebounding and D." #pos
predict_sentiment(test_sentence5)
test_sentence6 = "Kevin Durant is still better than lebron James." # pos
predict_sentiment(test_sentence6)