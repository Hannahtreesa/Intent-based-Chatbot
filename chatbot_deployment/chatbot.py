import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('punkt')

# Example data for training
patterns = ["Hello", "Hi", "How are you?", "What is your name?", "Goodbye", "See you later"]
tags = ["greeting", "greeting", "greeting", "question", "goodbye", "goodbye"]

# Encode tags
label_encoder = LabelEncoder()
tags_encoded = label_encoder.fit_transform(tags)

# Vectorize patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train the model
model = MultinomialNB()
model.fit(X, tags_encoded)

# Save the trained model, vectorizer, and label encoder
with open('chatbot_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

print("Model, vectorizer, and label encoder saved successfully!")
