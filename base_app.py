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
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
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
#raw = pd.read_csv("resources/train.csv")
train_data = pd.read_csv('https://raw.githubusercontent.com/rufusseopa/classification-predict-streamlit-template/master/Data/train.csv')
test_data = pd.read_csv('https://raw.githubusercontent.com/rufusseopa/classification-predict-streamlit-template/master/Data/test.csv')

def clean_text(raw): 
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

train_data['message'] = train_data['message'].apply(clean_text)
test_data['message'] = test_data['message'].apply(clean_text)






# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
    
	html_temp = """
	<div style="background-color:tomato;padding:10px">
	<h1 style="color:white;text-align:center;">Tweet Classifier App </h2>
	</div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer (**_Climate Change Sentiment_**)")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction",'Data Insights','About the app','Video about climate change','What is Climate Change?']
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "About the app":
		st.info("General Information about the app")
        

		# You can read a markdown file from supporting resources folder
		st.markdown("Below we can see an extract of the dataset which contains a column for the sentiment and another for the tweet. This dataset was used to develop the app")
        
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(train_data[['sentiment', 'message']]) # will write the df to the page
            
		st.subheader('How to use the tweet classifier')
		st.markdown(open('resources/info.md').read())       
            
            
    # Building out the "Data Insights" page
	if selection == "Data Insights":
		st.info("Insights from the dataset")
		# You can read a markdown file from supporting resources folder
		st.markdown("The following page contains visuals related to the dataset")

		st.subheader("View visuals")
        
        #product pie chart showing distribution of tweets
		if st.checkbox('Display distribution of tweets'):
			train_data['sentiment'].value_counts().plot(kind='pie',title='Distribution of classes',autopct='%1.1f%%')
			st.pyplot()
            
        #product bar chart showing len of tweets
		if st.checkbox('Display length of tweets'):
            # Separate the classes
			news = train_data[train_data['sentiment']==2]
			pro = train_data[train_data['sentiment']==1]
			neutral = train_data[train_data['sentiment']==0]
			anti = train_data[train_data['sentiment']==-1]
			st.subheader('Length of tweets')
			fig, axs = plt.subplots(2, 2, figsize=(11,7))

			axs[0, 0].hist(pro.message.str.len(),bins=50,label='pro',color='grey')
			axs[0, 0].set_title('pro')

			axs[1, 0].set_title('news')
			axs[1, 0].hist(news.message.str.len(),bins=50,label='news',color='lime')

			axs[0, 1].set_title('neutral')
			axs[0, 1].hist(neutral.message.str.len(),bins=50,label='neutral',color='brown')

			axs[1, 1].set_title('anti')
			axs[1, 1].hist(anti.message.str.len(),bins=50,label='anti',color='blue')

			for ax in axs.flat:
				ax.set(xlabel='length of tweets', ylabel='number of tweets')

            # Hide x labels and tick labels for top plots and y ticks for right plots.
			for ax in axs.flat:
				ax.label_outer() 
			st.pyplot()    

        
        #product wordclouds                                    
		if st.checkbox('Display wordclouds'):
			pro_tweets = train_data[train_data['sentiment'] == 1]
			all_words = ''.join([label for label in pro_tweets['message']])
			wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, max_words=50).generate(all_words)
			st.subheader("Pro climate change")
			plt.imshow(wordcloud, interpolation="bilinear")
			plt.axis('off')
			plt.show()
			st.pyplot()
            
			pro_tweets = train_data[train_data['sentiment'] == -1]
			all_words = ''.join([label for label in pro_tweets['message']])
			wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, max_words=50).generate(all_words)
			st.subheader("Negative sentiments")
			plt.imshow(wordcloud, interpolation="bilinear")
			plt.axis('off')
			plt.show()
			st.pyplot() 
            
			pro_tweets = train_data[train_data['sentiment'] == 0]
			all_words = ''.join([label for label in pro_tweets['message']])
			wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, max_words=50).generate(all_words)
			st.subheader("Neutral sentiment")
			plt.imshow(wordcloud, interpolation="bilinear")
			plt.axis('off')
			plt.show()
			st.pyplot()  
            
			pro_tweets = train_data[train_data['sentiment'] == 2]
			all_words = ''.join([label for label in pro_tweets['message']])
			wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, max_words=50).generate(all_words)
			st.subheader("Factual news")
			plt.imshow(wordcloud, interpolation="bilinear")
			plt.axis('off')
			plt.show()
			st.pyplot()               
            

    # Building out the "View video" page
	if selection == "Video about climate change":
		st.info("Educational video about climate change")
		# You can read a markdown file from supporting resources folder
		#st.markdown("The following page contains a youtube video about climate change")

		st.subheader("This page contains a youtube video explaing what climate change is and the effects of it")
		if st.checkbox('View video'): # data is hidden if box is unchecked
			st.video('https://www.youtube.com/watch?v=ifrHogDujXw&t=12s')
            
    # Building out the "What is climate change" page
	if selection == "What is Climate Change?":
		st.info("Brief overview of climate change")
		# You can read a markdown file from supporting resources folder
		st.subheader("The following page contains information about what climate change is and the effects of it")
		st.markdown(open('resources/climate change info.md').read())
		image = Image.open('resources/polar bear.png')
		st.image(image, caption='Effects on wildlife')




	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text below to classify it","Type Here")
        
        #Create sidebar for user to choose model
		model_opt = ["Logistic Regression","Linear SVC","Kernel SVM"]
		select_model = st.sidebar.selectbox("Choose Model", model_opt)        

		if st.button("Classify"):
            #Convert every tweet to be lower case, we do this to reduce some noise.
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
            
            #remove extra space
			#tweet_text = tweet_text.split().str.join(" ")
            
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
            
			if select_model == "Logistic Regression":                
				predictor = joblib.load(open(os.path.join("resources/logreg.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

			if select_model == "Linear SVC":                
				predictor = joblib.load(open(os.path.join("resources/linsvc1.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])
   
			if select_model == "Kernel SVM":                
				predictor = joblib.load(open(os.path.join("resources/kernelsvm.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])            
            
            
			#predictor = joblib.load(open(os.path.join("resources/kernelsvm.pkl"),"rb"))
			#prediction = predictor.predict([tweet_text])
            
			st.success("Text Categorized as: {}".format(prediction))                        

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			x =(prediction)
			if x==1:
				st.success('Tweet supports man-made climate change')
			elif x==2:
				st.success('Tweet links to news about climate change')
			elif x==-1:
				st.success('Tweet does not support man-made climate change')
			else:
				st.success('Tweet neither supports nor refutes the belief of man-made climate change')                


            
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
