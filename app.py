# run by this streamlit run app.py
# Importing necessary libraries
import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

# Initializing Porter Stemmer and TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

# Loading pre-trained vectorizer and model from pickled files
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# Function for stemming the input content
def stemming(content):
    # Removing non-alphabetic characters and converting to lowercase
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    # Splitting into words
    con = con.split()
    # Stemming and removing stopwords
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    # Joining the words back into a string
    con = ' '.join(con)
    return con

# Function for predicting fake news
def fake_news(news):
    # Applying stemming to the input news
    news = stemming(news)
    # Transforming the input using the pre-trained vectorizer
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    # Making predictions using the pre-trained model
    prediction_prob = load_model.predict_proba(vector_form1)
    prediction_class = load_model.predict(vector_form1)
    return prediction_prob, prediction_class

# Main section
if __name__ == '__main__':
    # Setting up the Streamlit app title and input section
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("Predict News")

    # Handling prediction button click
    if predict_btt:
        # Making a prediction using the input news content
        prediction_prob, prediction_class = fake_news(sentence)
        
        # Displaying the predicted class and probability score
        st.subheader('Prediction:')
        if prediction_class == [0]:
            st.success('The News Is Reliable')
        elif prediction_class == [1]:
            st.warning('The News Is Unreliable')
        
        # Displaying the probability scores
        st.subheader('Prediction Probability:')
        st.write(f'Reliable: {prediction_prob[0][0]:.4f}')
        st.write(f'Unreliable: {prediction_prob[0][1]:.4f}')

        # Plotting a bar chart for better visualization
        prob_df = pd.DataFrame({'Class': ['Reliable', 'Unreliable'], 'Probability': [prediction_prob[0][0], prediction_prob[0][1]]})
        st.subheader('Prediction Probability Chart:')
        st.bar_chart(prob_df.set_index('Class'))

        
        # Displaying the confusion matrix
        st.subheader('Confusion Matrix:')
        y_true = [0] 
        cm = confusion_matrix(y_true, prediction_class)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        st.pyplot(fig)
        
        # Displaying the word cloud
        st.subheader('Word Cloud:')
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentence)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)