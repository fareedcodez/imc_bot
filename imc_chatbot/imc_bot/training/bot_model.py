import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class ChatbotTrainer:
    def __init__(self, training_data_path):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.label_encoder = LabelEncoder()
        self.training_data = self.load_training_data(training_data_path)
        
    def load_training_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words and token.isalnum()]
        
        return " ".join(tokens)
    
    def prepare_training_data(self):
        texts = []
        labels = []
        
        for intent in self.training_data['intents']:
            for pattern in intent['patterns']:
                texts.append(self.preprocess_text(pattern))
                labels.append(intent['tag'])
        
        # Convert text to TF-IDF features
        X = self.vectorizer.fit_transform(texts).toarray()
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        y = tf.keras.utils.to_categorical(y)
        
        return X, y
    
    def create_model(self, input_shape, num_classes):
        model = Sequential([
            Dense(128, input_shape=(input_shape,), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, model_save_path):
        X, y = self.prepare_training_data()
        
        model = self.create_model(X.shape[1], y.shape[1])
        
        # Train the model
        model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Save the model, vectorizer, and label encoder
        model.save(model_save_path)
        return model

class Chatbot:
    def __init__(self, model_path, training_data_path):
        self.model = tf.keras.models.load_model(model_path)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.label_encoder = LabelEncoder()
        self.training_data = self.load_training_data(training_data_path)
        self.fit_vectorizer()
        self.fit_label_encoder()
    
    def load_training_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def fit_vectorizer(self):
        texts = []
        for intent in self.training_data['intents']:
            for pattern in intent['patterns']:
                texts.append(self.preprocess_text(pattern))
        self.vectorizer.fit(texts)
    
    def fit_label_encoder(self):
        labels = []
        for intent in self.training_data['intents']:
            labels.append(intent['tag'])
        self.label_encoder.fit(labels)
    
    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words and token.isalnum()]
        
        return " ".join(tokens)
    
    def get_response(self, text):
        # Preprocess and vectorize input text
        processed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([processed_text]).toarray()
        
        # Predict intent
        predictions = self.model.predict(text_vector)
        predicted_tag = self.label_encoder.inverse_transform([np.argmax(predictions)])
        
        # Find matching intent and select random response
        for intent in self.training_data['intents']:
            if intent['tag'] == predicted_tag[0]:
                return np.random.choice(intent['responses'])
        
        return "I'm not sure how to respond to that."