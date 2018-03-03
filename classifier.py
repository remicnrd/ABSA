
import io
import keras
import spacy
import pandas as pd
from keras.models import load_model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

class Classifier:
    """The Classifier"""
    vocab_size = 3000

    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        train = pd.read_csv(trainfile, sep="\t", header=None,
        	names = ["polarity", "aspect_category", "aspect_term", "at_location", "sentence"])

        # We use a tokenizer for text features, and an encoder for categorical features
        tokenize = Tokenizer(num_words=vocab_size)
        tokenize.fit_on_texts(train.sentence)
        with open('tokenizer.pickle', 'wb') as handle:
        	pickle.dump(tokenizer, handle)
        label_encoder = LabelEncoder()

        sentence_tokenized = pd.DataFrame(tokenize.texts_to_matrix(train.sentence))
        aspect_tokenized = pd.DataFrame(tokenize.texts_to_matrix(train.aspect_term))

        integer_category = label_encoder.fit_transform(train.aspect_category)
        one_hot_category = pd.DataFrame(to_categorical(integer_category))
        integer_polarity = label_encoder.fit_transform(train.polarity)
        one_hot_polarity = pd.DataFrame(to_categorical(integer_polarity))

        # Define X and y for network input and output
        X = pd.concat([one_hot_category, aspect_tokenized, sentence_tokenized], axis=1)
        y = one_hot_polarity

        # Define the Neural Network
        model = Sequential()
        model.add(Dense(512, input_shape=(6012,)))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', 
        	optimizer='adam', 
        	metrics=['accuracy'])

        model.fit(X, Y, epochs=2, verbose=1)
        model.save('model.h5')




    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test = pd.read_csv(datafile, sep="\t", header=None,
        	names = ["polarity", "aspect_category", "aspect_term", "at_location", "sentence"])

        # Data preprocessing
        with open('tokenizer.pickle', 'rb') as handle:
        	tokenizer = pickle.load(handle)
        label_encoder = LabelEncoder()

        sentence_tokenized = pd.DataFrame(tokenize.texts_to_matrix(test.sentence))
        aspect_tokenized = pd.DataFrame(tokenize.texts_to_matrix(test.aspect_term))
        
        integer_category = label_encoder.fit_transform(test.aspect_category)
        one_hot_category = pd.DataFrame(to_categorical(integer_category))

        X = pd.concat([one_hot_category, aspect_tokenized, sentence_tokenized], axis=1)

        model = load_model('model.h5')
        predictions = model.predict_classes(X_test,)
        return label_encoder.inverse_transform(predictions)


