import io
import keras
import spacy
import pickle
import pandas as pd
from keras.models import load_model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

def clean_text(df, colname):
    """Lowercase, remove stopwords and lemmatize the text column"""
    nlp = spacy.load('en')
    lower_sentence = []
    clean_sentence = []

    for index, row in df.iterrows():
        lower_sentence.append(row[colname].lower())
    df[colname] = lower_sentence

    for doc in nlp.pipe(df[colname].astype('unicode').values):
        if doc.is_parsed:
            clean_sentence.append(' '.join([n.lemma_ for n in doc if (not n.is_stop and not n.is_punct)])) 
        else:
            # To keep the same number of entries
            clean_sentence.append('')    
    df[colname] = clean_sentence

class Classifier:
    """The Classifier"""
  
    #############################################

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        train = pd.read_csv(trainfile, sep="\t", header=None,
        	names = ["polarity", "aspect_category", "aspect_term", "at_location", "sentence"])

        # Text preprocessing for sentences
        clean_text(train, 'sentence')

        # Create and fit tokenizer with limited word number, save it for prediction
        vocab_size = 3000
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(train.sentence)
        with open('tokenizer.pickle', 'wb') as handle:
        	pickle.dump(tokenizer, handle)

        # Label encoders for categorical variables
        label_encoder_category = LabelEncoder()
        label_encoder_polarity = LabelEncoder()


        # Define the BoW vectors using the same matrix
        sentence_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(train.sentence))
        aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(train.aspect_term))

        # Format categorical variables to one hot
        integer_category = label_encoder_category.fit_transform(train.aspect_category)
        one_hot_category = pd.DataFrame(to_categorical(integer_category))
        integer_polarity = label_encoder_polarity.fit_transform(train.polarity)
        one_hot_polarity = pd.DataFrame(to_categorical(integer_polarity))

        # Save the fitted label encoder for prediction decoding
        with open('label_encoder_polarity.pickle', 'wb') as handle:
        	pickle.dump(label_encoder_polarity, handle)

        # Define predictors and dependant variable
        X = pd.concat([one_hot_category, aspect_tokenized, sentence_tokenized], axis=1)
        y = one_hot_polarity

        # Create, compile, fit and save the Neural Network
        model = Sequential()
        model.add(Dense(512, input_shape=(6012,)))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', 
        	optimizer='adam', 
        	metrics=['accuracy'])
        model.fit(X, y, epochs=2, verbose=1)
        model.save('model.h5')




    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test = pd.read_csv(datafile, sep="\t", header=None,
        	names = ["polarity", "aspect_category", "aspect_term", "at_location", "sentence"])

        # Text preprocessing for sentences
        clean_text(test, 'sentence')

        # Use fitted tokenizer  to keep the same BoW vectors
        with open('tokenizer.pickle', 'rb') as handle:
        	tokenizer = pickle.load(handle)

        # Use fitted label_Encoder to decode predictions
        with open('label_encoder_polarity.pickle', 'rb') as handle:
            label_encoder_polarity = pickle.load(handle)

        label_encoder = LabelEncoder()

        # Define the BoW vectors using the same matrix
        sentence_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(test.sentence))
        aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(test.aspect_term))
        
        # Format categorical variables to one hot
        integer_category = label_encoder.fit_transform(test.aspect_category)
        one_hot_category = pd.DataFrame(to_categorical(integer_category))

        X = pd.concat([one_hot_category, aspect_tokenized, sentence_tokenized], axis=1)

        # Load the model weights and architecture, predict for new data
        model = load_model('model.h5')
        predictions = model.predict_classes(X,)
        return label_encoder_polarity.inverse_transform(predictions)


