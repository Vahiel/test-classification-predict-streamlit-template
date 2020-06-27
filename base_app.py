"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')
import pickle
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Clssifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":

		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			tweet_text = tweet_text.lower()
            #Remove stop words
			def stop_words(text):
				word = text.split()
                #Remove stop words
				stop_word = set(stopwords.words("english"))
				remove_stop = [w for w in word if w not in stop_word]
				free_stop = " ".join(remove_stop)
				return free_stop
			tweet_text = stop_words(tweet_text)
            
			spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–","0123456789"]
			for char in spec_chars:
				tweet_text = tweet_text.replace(char, ' ')
			def clean_ing(raw): 
			# Remove link
				raw = re.sub(r'http\S+', '', raw)
                # Remove "RT"
				raw = re.sub('RT ', '', raw)
                # Remove unexpected artifacts
				raw = re.sub(r'â€¦', '', raw)
				raw = re.sub(r'…', '', raw)
				raw = re.sub(r'â€™', "'", raw)
				raw = re.sub(r'â€˜', "'", raw)
				raw = re.sub(r'\$q\$', "'", raw)
				raw = re.sub(r'&amp;', "and", raw)
				words = raw.split()  

				return( " ".join(words))
            
			tweet_text = clean_ing(tweet_text)                
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/kernelsvm.pkl"),"rb"))
			prediction = predictor.predict([tweet_text])

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
