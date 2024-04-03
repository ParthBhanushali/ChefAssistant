import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gradio as gr

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

x = pd.read_csv('train.csv')

zucchini_recipes = x[x['recipe_id'].str.contains('beef')]
zucchini_recipes#['context_body'].iloc[0]

class FAQChatbot:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_model = self.train_word2vec()
        self.tfidf_vectorizer, self.tfidf_matrix = self.compute_tfidf()

    def load_data(self, file_path):
        try:
            # Load CSV data
            df = pd.read_csv(file_path)

            # Remove rows with NaN values
            df = df.dropna()

            # Replace dashes with spaces in recipe_id column
            df['recipe_id'] = df['recipe_id'].str.replace('-', ' ')

            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File '{file_path}' not found.")
        except ValueError as ve:
            raise ValueError(f"Error: Input data contains NaN values. Details: {ve}")

    def preprocess_text(self, text):
        if text:
            words = word_tokenize(text.lower())

            # Remove stopwords and lemmatize
            words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
            return words
        else:
            return []

    def compute_tfidf(self):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['recipe_id'])
        return tfidf_vectorizer, tfidf_matrix

    def train_word2vec(self):
        sentences = [self.preprocess_text(sentence) for sentence in self.data['recipe_id']]
        model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
        return model

    def find_most_relevant_response(self, input_question):
        try:
            input_preprocessed = self.preprocess_text(input_question)

            # Compute TF-IDF similarity
            input_vector = self.tfidf_vectorizer.transform([input_question])
            tfidf_similarities = cosine_similarity(input_vector, self.tfidf_matrix)
            tfidf_most_similar_index = np.argmax(tfidf_similarities)
            tfidf_response = self.data['context_body'].iloc[tfidf_most_similar_index]

            # Compute Word2Vec similarity if input words are present in the vocabulary
            word2vec_similarities = []
            for sentence in self.data['recipe_id']:
                sentence_preprocessed = self.preprocess_text(sentence)
                if all(token in self.word2vec_model.wv.key_to_index for token in sentence_preprocessed) and all(token in self.word2vec_model.wv.key_to_index for token in input_preprocessed):
                    input_embedding = np.mean([self.word2vec_model.wv[token] for token in input_preprocessed], axis=0)
                    sentence_embedding = np.mean([self.word2vec_model.wv[token] for token in sentence_preprocessed], axis=0)
                    similarity = cosine_similarity(input_embedding.reshape(1, -1), sentence_embedding.reshape(1, -1))[0][0]
                    word2vec_similarities.append(similarity)
                else:
                    word2vec_similarities.append(-1)  # Placeholder value for words not in vocabulary

            word2vec_most_similar_index = np.argmax(word2vec_similarities)
            word2vec_response = self.data['context_body'].iloc[word2vec_most_similar_index]

            # Decide which response to return based on a combination of both scores
            tfidf_weight = 0.9
            word2vec_weight = 0.1
            combined_score = tfidf_weight * tfidf_similarities[0, tfidf_most_similar_index] + word2vec_weight * word2vec_similarities[word2vec_most_similar_index]
            if combined_score >= 0.5:  # Adjust threshold as needed
                return tfidf_response
            else:
                return word2vec_response
        except ValueError as ve:
            # Handle NaN values in input_question
            raise ValueError(f"Error: Input contains NaN. Details: {ve}")

# Load FAQChatbot
faq_chatbot = FAQChatbot('train.csv')

def chatbot(input_question):
    try:
        response = faq_chatbot.find_most_relevant_response(input_question)
        return response
    except ValueError as ve:
        return f"Error: {ve}"

chatbot_interface = gr.Interface(
    fn=chatbot,
    inputs=['text'],
    outputs="text",
    title="Advanced FAQ Chatbot",
    description="Ask any question related to the provided dataset. This version incorporates multiple techniques to improve accuracy, including TF-IDF and Word2Vec embeddings.",
    examples=[
        ["What is the recipe for caramel dumplings?"],
        ["How do I make zucchini bread?"],
        ["How to make zucchini pizza?"],

    ],
    allow_flagging = False,
    theme="black"
)

chatbot_interface.launch(share = True)



