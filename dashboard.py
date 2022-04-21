import pandas as pd
import numpy as np
import plotly.express as px
import pygooglenews as pgn
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Necessary during first run of the project
# nltk.download('stopwords')
# nltk.download('SentimentIntensityAnalyzer')

requirements = """
How to install the required libraries in a conda environment:

conda create --name itpproject python=3.7
conda activate itpproject
conda install -c anaconda feedparser=5.2.1
pip install pygooglenews
conda install plotly
pip install streamlit
pip install matplotlib
pip install pandas
pip install numpy
pip install nltk

Other helpful conda commands:

conda env remove --name itpproject
conda uninstall python
conda install python=3.7

How to run the dashboard.py file:

1. Give full disk access to the terminal (in the Security & Privacy settings).
2. Open the terminal and change the directory to the project folder.
3. Activate the respective environment.
4. Run the following command: streamlit run dashboard.py

"""


COUNTRY = 'Ukraine'
NEWS_POPULATION_THRESHOLD = 500_000 # population threshold which cities shall be considered
NEWS_RETROSPECT_THRESHOLD = 7 # The number of days of which news in the past should be considered.
SAMPLE_NUMBER = 5

NEGATIVE_WORDS = ['war', 'russia', 'russian', 'bomb',
'bombs', 'attack', 'missile', 'fire', 'soldier', 'tank',
'weapon', 'deaths', 'death', 'destruction','zelenskyy']

st.set_page_config(page_title=f'{COUNTRY} News Dashboard', 
					page_icon='🌍')

st.title(f'{COUNTRY} News Dashboard')

#########################################################################################################
# Declaration of relevant functions.																	#
#########################################################################################################

@st.cache(suppress_st_warning=True)
def get_world_cities():
	"""Returns a pandas dataframe of all cities with more than NEWS_POPULATION_THRESHOLD citizens.
	The .csv file has been retrieved from https://simplemaps.com/data/world-cities."""

	world_cities = pd.read_csv('resources/worldcities.csv')

	cities = world_cities.loc[(world_cities['country'] == COUNTRY) & (world_cities['population'] >= NEWS_POPULATION_THRESHOLD)]
	
	return cities

cities = get_world_cities()

@st.cache(suppress_st_warning=True)
def get_news_per_city(city_names = cities['city'], search_in_title = True):
	"""Returns a dictionary where the city names represent the keys and the news titles about the city the values."""

	text_placeholder = st.empty()
	text_placeholder.write(f'The first run of the app may take a while since the programm needs to fetch the latest news articles about all Ukrainian cities with more than {NEWS_POPULATION_THRESHOLD:,} citizens. Please be patient.')
	
	city_placeholder = st.empty()

	num_cities = len(city_names)
	progress_bar = st.progress(0)
	counter = 0

	if search_in_title:
		t = 'intitle:'
	else:
		t= ''

	gn = pgn.GoogleNews()
	news = {}
	for city in city_names:

		population_current_city = int(cities.loc[cities['city'] == city, ['population']]['population'])

		city_placeholder.markdown(f'Current city: {city}  \n Population: {population_current_city:,}')

		counter += 1
		progress_bar.progress(counter / num_cities)

		search = gn.search(f'{t}{city}', when=f'{NEWS_RETROSPECT_THRESHOLD}d')

		temp_title_list = []

		for element in search['entries']:
			temp_title_list.append(element['title'])

		news[city] = temp_title_list
	
	progress_bar.empty()
	text_placeholder.empty()
	city_placeholder.empty()

	return news

news = get_news_per_city()

@st.cache(suppress_st_warning=True)
def tokenize_remove_stop_words(news_dict = news):
	"""Takes a dictionary of lists, removes stopwords and cleans the text.
	Required to count most frequent words."""

	token_news = {}
	stop_words = set(stopwords.words('english'))

	for city in news_dict.keys():
		token_news[city] = []

		for element in news_dict[city]:
			words = element.lower().split()
			
			token_news[city].append([word for word in words if word not in stop_words])

	return token_news

token_news = tokenize_remove_stop_words()

def plot_map():

	fig = px.scatter_mapbox(cities[cities['population'] >= plot_threshold],
							lat='lat',
							lon='lng',
							size='population',
							zoom=5,
							color='compound',
							color_continuous_scale = 'Bluered_r',
							range_color = [-1,1], # remove if colors should be more distinguishable
							hover_name='city',
							hover_data=dict(lat=False,
											lng=False,
											population=':,')) #set thousands delimiter

	fig.update_layout(mapbox_style='stamen-terrain')
	fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
	fig.update_layout(title_text="Ukraine")
	#fig.show()
	
	return fig

def get_sentiment_per_news(news_dict = news):

	nltk.download('vader_lexicon')

	sia = SentimentIntensityAnalyzer()

	sentiment_news = {}

	for city in news_dict.keys():

		sentiment_news[city] = []

		for title in news_dict[city]:

			sentiment_news[city].append({'title': title,
										'compound': sia.polarity_scores(title)['compound']})

	return sentiment_news

sentiment_news = get_sentiment_per_news()

def get_average_sentiment_per_city(sentiment_news_dict = sentiment_news):

	average_compund_per_city = {}

	for city in sentiment_news_dict.keys():

		temp_list_of_compounds = []

		for title in sentiment_news[city]:
			temp_list_of_compounds.append(float(title['compound']))

		if len(temp_list_of_compounds) > 0:
			average_compund_per_city[city] = [sum(temp_list_of_compounds) / len(temp_list_of_compounds)]
		else:
			average_compund_per_city[city] = [0]
		
		average_compund_per_city[city].append(len(temp_list_of_compounds))

	average_compund_per_city = pd.DataFrame.from_dict(average_compund_per_city, orient='index').reset_index()

	average_compund_per_city.columns = ('city', 'compound', 'number_of_articles')

	return average_compund_per_city

average_compund_per_city = get_average_sentiment_per_city()

#########################################################################################################
# Content of the dashboard.																				#
#########################################################################################################

cities = cities.merge(average_compund_per_city, left_on='city', right_on='city')


plot_threshold = st.slider('Population Threshold', 
							min_value = NEWS_POPULATION_THRESHOLD,
							max_value=1000_000,
							step=50_000,
							value = NEWS_POPULATION_THRESHOLD)

st.write(f'This map shows all {COUNTRY} cities with a population larger than {plot_threshold:,}. The size of the bubbles correlates with the population whereas the color correlates with the average polarity (positive vs. negative) of the articles where 1 represents extremely positive news and -1 represents extremely negative news.')

st.plotly_chart(plot_map())

if st.button('View DataFrame'):
	st.write('The DataFrame is sorted by average word polarity.')
	st.write(cities.sort_values('compound'))
	st.button('Close DataFrame')


st.subheader('News Samples')

st.write("""In this section you can find a sample of current news about the city you select.""")

selected_city = st.selectbox(label='About which city do you want to know more?', options=cities['city'])
selected_city_news = pd.DataFrame(news[selected_city])

news_available = True
if len(selected_city_news) == 0:
	news_available = False
else:
	selected_city_news.columns = [f'News about {selected_city}']


if len(selected_city_news) >= SAMPLE_NUMBER:	
	selected_city_news_sample = selected_city_news.sample(SAMPLE_NUMBER)
else:
	selected_city_news_sample = selected_city_news

if news_available:
	for i, headline in enumerate(selected_city_news_sample[f'News about {selected_city}']):
		st.write(f'{i + 1}. {headline}')
else:
	st.write('Unfortunately, there are no news available for the city you selected.')
	
st.subheader(f'Word Finder')

#########################################################################################################
# TBD																									#
#########################################################################################################
# Search for number of occurrences of word.




#########################################################################################################
# Page Test																								#
#########################################################################################################

st.subheader(f'Page Test')

if 'count' not in st.session_state:
    st.session_state.count = 0

increment = st.button('Increment')
if increment:
    st.session_state.count += 1

st.write('Count = ', st.session_state.count)

title = st.text_input('Movie title', 'Life of Brian')