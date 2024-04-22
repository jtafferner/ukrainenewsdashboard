import pandas as pd
import numpy as np
import plotly.express as px
from modules import googlenews as gnews
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import datetime

# downloading the required nltk packages. if the package is already downloaded, it will not be downloaded again.
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')

# declaration of dashboard constants
COUNTRY = 'Ukraine' # country to be observed
NEWS_POPULATION_THRESHOLD = 50_000 # population threshold which cities shall be considered
NEWS_RETROSPECT_THRESHOLD = 7 # the number of days of which news in the past should be considered.
SAMPLE_NUMBER = 5 # number of news samples to be shown

st.set_page_config(page_title=f'{COUNTRY} News Dashboard', page_icon='🌍', initial_sidebar_state='collapsed')
st.title(f'{COUNTRY} News Dashboard')

#########################################################################################################
# Declaration of relevant functions.																	#
#########################################################################################################

with st.sidebar:

	country_list = pd.read_csv('resources/worldcities.csv')
	idx = int(country_list['country'].unique().tolist().index('Ukraine'))
	st.write('**App Configurations**')
	COUNTRY = st.selectbox(label='Which country do you wish to observe?', options=country_list['country'].unique(), index=idx)

@st.cache(suppress_st_warning=True)
def get_world_cities():
	"""Returns a pandas dataframe of all cities with more than NEWS_POPULATION_THRESHOLD citizens.
	The .csv file has been retrieved from https://simplemaps.com/data/world-cities."""

	world_cities = pd.read_csv('resources/worldcities.csv')
	cities = world_cities.loc[(world_cities['country'] == COUNTRY) & (world_cities['population'] >= NEWS_POPULATION_THRESHOLD)]
	
	return cities, world_cities

cities, world_cities = get_world_cities()

@st.cache(suppress_st_warning=True)
def get_news_per_city(city_names = cities['city'], search_in_title = True):
	"""Returns a dictionary where the city names represent the keys and the news titles about the city the values."""

	text_placeholder = st.empty()
	text_placeholder.write(f'The first run of the app may take a while since the programm needs to fetch the latest news articles about all cities with more than {NEWS_POPULATION_THRESHOLD:,} citizens. Please be patient.')
	city_placeholder = st.empty()

	num_cities = len(city_names)
	progress_bar = st.progress(0)
	counter = 0

	if search_in_title:
		t = 'intitle:'
	else:
		t= ''

	gn = gnews.GoogleNews()
	news = {}
	try:
		
		for city in city_names:

			population_current_city = int(cities.loc[cities['city'] == city, ['population']]['population'])
			city_placeholder.markdown(f'Current city: {city}  \n Population: {population_current_city:,}')

			counter += 1
			progress_bar.progress(counter / num_cities)

			search = gn.search(f'{t}{city}', when=f'{NEWS_RETROSPECT_THRESHOLD}d')

			temp_title_list = []
			for element in search['entries']:
				
				if element['title'].count('-') > 0:
					
					cutoff = element['title'].rfind('-') - 1
					title = element['title'][:cutoff]
					temp_title_list.append({'title': title, 'link': element['link'], 'pubdate': element['pubdate']})
				
				else:
					temp_title_list.append({'title': element['title'], 'link': element['link'], 'pubdate': element['pubdate']})

			news[city] = temp_title_list

	except:
		st.write('WARNING: Streamlit is not able to handle the amount of news data required to display the full map. Hence, only the cities that have been analyzed so far are visible on the map and in the rest of the analyses down below.')
	
	progress_bar.empty()
	text_placeholder.empty()
	city_placeholder.empty()
	
	date_time = datetime.datetime.now().replace(microsecond=0)

	return news, date_time

news, date_time = get_news_per_city()

def tokenize_remove_stop_words(news_dict = news, lemmatize = True):
	"""Takes a dictionary of lists, removes stopwords and cleans the text.
	Required to count most frequent words."""

	token_news = {}
	stop_words = set(stopwords.words('english'))
	forbidden_words = ('news', 'russia', 'ukraine', 'ukrainian', 'russian')
	if lemmatize: lemmatizer = WordNetLemmatizer()

	for city in news_dict.keys():
		token_news[city] = []

		for element in news_dict[city]:
			words = element['title'].lower().split()

			if lemmatize:
				words = [lemmatizer.lemmatize(word) for word in words]

			token_news[city].append([word for word in words if (word not in stop_words) and (word.isalpha()) and (len(word) > 1) and (word != city.lower()) and (word not in ['ukraine', 'ukrainian']) and (word not in forbidden_words)])

	return token_news

token_news = tokenize_remove_stop_words()

def plot_map(size='population', color='polarity', range_color=[-1,1], color_continuous_scale='Bluered_r'):

	fig = px.scatter_mapbox(cities[cities['population'] >= plot_threshold],
							lat='lat',
							lon='lng',
							size=size,
							zoom=5,
							color=color,
							color_continuous_scale = color_continuous_scale,
							range_color = range_color, # remove if colors should be more distinguishable
							hover_name='city',
							hover_data=dict(lat=False,
											lng=False,
											population=':,',
											number_of_articles=':,')) #set thousands delimiter

	fig.update_layout(mapbox_style='open-street-map')
	fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
	fig.update_layout(title_text="Ukraine")
	
	return fig

def get_sentiment_per_news(news_dict = news):

	sia = SentimentIntensityAnalyzer()
	sentiment_news = {}

	for city in news_dict.keys():
		sentiment_news[city] = []

		for title in news_dict[city]:
			sentiment_news[city].append({'title': title['title'], 'compound': sia.polarity_scores(title['title'])['compound']})

	return sentiment_news

sentiment_news = get_sentiment_per_news()

def get_average_sentiment_per_city(sentiment_news_dict = sentiment_news):

	average_compund_per_city = {}

	for city in sentiment_news_dict.keys():
		temp_list_of_compounds = []

		for title in sentiment_news[city]:
			temp_list_of_compounds.append(float(title['compound']))

		if len(temp_list_of_compounds) > 0:
			average_compund_per_city[city] = [round(sum(temp_list_of_compounds) / len(temp_list_of_compounds), 2)]
		else:
			average_compund_per_city[city] = [0]
		
		average_compund_per_city[city].append(len(temp_list_of_compounds))

	average_compund_per_city = pd.DataFrame.from_dict(average_compund_per_city, orient='index').reset_index()
	average_compund_per_city.columns = ('city', 'polarity', 'number_of_articles')

	return average_compund_per_city

average_compund_per_city = get_average_sentiment_per_city()

def get_occurence_of_word_per_city(word = 'Missile', news_dictionary = news):

	word = word.lower()
	occurrence_per_city = {}

	for city in news_dictionary.keys():
		count = 0

		for headline in news_dictionary[city]:
			count += headline['title'].lower().count(word)
		
		if len(news_dictionary[city]) != 0:
			occurrence_per_city[city] = round(count / len(news_dictionary[city]), 2)
		else:
			occurrence_per_city[city] = 0

	occurrence_per_city = pd.DataFrame.from_dict(occurrence_per_city, orient='index').reset_index()
	occurrence_per_city.columns = ('city', 'word_count')

	return occurrence_per_city

def word_frequency_per_city():
	
	shallow_token_news = {}
	for city in token_news.keys():	
		shallow_token_news[city] = []
		
		for headline in token_news[city]:
			for word in headline:
				if word.lower() not in city.lower().split() and word.lower() not in COUNTRY.lower().split():

					shallow_token_news[city].append(word)
	
	word_frequencies = {}
	for city in shallow_token_news.keys():
		word_frequencies[city] = {}
		
		for word in shallow_token_news[city]:
			if word in word_frequencies[city].keys():
				word_frequencies[city][word] += 1
			else:
				word_frequencies[city][word] = 1

	for city in word_frequencies.keys():

		for word in word_frequencies[city].keys():

			word_frequencies[city][word] /= len(token_news[city])
			word_frequencies[city][word] = round(word_frequencies[city][word], 2)
			
	
	return word_frequencies

def plot_bar(word_frequency_dict = word_frequency_per_city(), color_word = 'Missile'):

	PLOT_NUMBER = 15

	city_dict = word_frequency_dict[selected_city_word]
	
	word_frequency_df = pd.DataFrame.from_dict(city_dict, orient='index').reset_index()
	word_frequency_df.columns = ('word', 'frequency')
	word_frequency_df.sort_values(by=['frequency'], inplace=True, ascending=False)
	limited_word_frequency_df = word_frequency_df.head(PLOT_NUMBER).reset_index()
	
	limited_word_frequency_df['category'] = [str(i) for i in limited_word_frequency_df.index]

	color_discrete_sequence = ['#005bbb'] * PLOT_NUMBER

	if color_word.lower() in list(limited_word_frequency_df['word']):
		index = limited_word_frequency_df.index[limited_word_frequency_df['word'] == color_word.lower()].tolist()[0]
		color_discrete_sequence[index] = '#ffd500'

	fig = px.bar(limited_word_frequency_df.head(PLOT_NUMBER), x='word', y='frequency', text='frequency', color='category', color_discrete_sequence=color_discrete_sequence)
	fig.update_layout(showlegend = False)

	return fig

#########################################################################################################
# Content of the dashboard.																				#
#########################################################################################################

try:
	cities = cities.merge(average_compund_per_city, left_on='city', right_on='city')
except:
	st.write('WARNING: Streamlit is not able to handle the amount of news data required to display the full map. Hence, only the cities that have been analyzed so far are visible on the map and in the rest of the analyses down below.')

plot_threshold = st.slider('Population Threshold', 
							min_value = NEWS_POPULATION_THRESHOLD,
							max_value=1000_000,
							step=50_000,
							value = NEWS_POPULATION_THRESHOLD)

st.write(f'This map shows all {COUNTRY} cities with a population larger than {plot_threshold:,}. The size of the bubbles correlates with the population whereas the color correlates with the average polarity (positive vs. negative) of the current news articles (last 7 days) from Google News where 1 represents extremely positive news and -1 represents extremely negative news. Try the slider above to display less cities on the map.')
st.plotly_chart(plot_map())

if st.button('View DataFrame'):
	st.write('The DataFrame is sorted by average word polarity.')
	st.write(cities.sort_values('polarity'))
	st.button('Close DataFrame')


st.subheader('News Samples')
st.write("""In this section you can find a sample of current news about the city you select.""")

selected_city = st.selectbox(label='About which city do you want to know more?', options=cities['city'], key=0)
selected_city_news = pd.DataFrame(news[selected_city])

news_available = True
if len(selected_city_news) == 0:
	news_available = False
else:
	selected_city_news.columns = [f'News about {selected_city}', 'Link', 'pubdate']


if len(selected_city_news) >= SAMPLE_NUMBER:	
	selected_city_news_sample = selected_city_news.sample(SAMPLE_NUMBER)
else:
	selected_city_news_sample = selected_city_news

if news_available:
	for i, headline in enumerate(selected_city_news_sample[f'News about {selected_city}']):
		link = selected_city_news_sample.loc[selected_city_news_sample[f'News about {selected_city}'] == headline, 'Link']
		pubdate = selected_city_news.loc[selected_city_news[f'News about {selected_city}'] == headline, 'pubdate'].values[0][:-13]
		st.write(f'{i + 1}. {headline} [[Link]]({list(link)[0]}) - Date: {pubdate}')
else:
	st.write('Unfortunately, there are no news available for the city you selected.')

if st.button('View all News'):
	st.subheader(f'List of all Current News about {selected_city}')
	st.write('This is a list of all current news available for the selected city. To collapse the list simply press the close button at the bottom or reload the page.')
	st.button('Collapse all News', key=0)
	
	if	f'News about {selected_city}' in selected_city_news.columns:
		for i, headline in enumerate(selected_city_news[f'News about {selected_city}']):
			link = selected_city_news.loc[selected_city_news[f'News about {selected_city}'] == headline, 'Link']
			pubdate = selected_city_news.loc[selected_city_news[f'News about {selected_city}'] == headline, 'pubdate'].values[0][:-13]
			st.write(f'{i + 1}. {headline} [[Link]]({list(link)[0]}) - Date: {pubdate}')
	else:
		st.write('Unfortunately, there are no news available for the city you selected.')
	st.button('Collapse all News', key=1)
	
#########################################################################################################
# Search for number of occurrences of word.																#
#########################################################################################################

st.subheader(f'Word Finder')
st.write('In this section you can search for a certain word that might appear in the current news about Ukraine and plot the frequency of the occurrence of the word per city. The brightness of the city bubbles correlates with the occurrence frequency. Down below, you can view all news including the respective word sorted on a city level.')
search = st.text_input(label='Which word do you wish to find?', value='Missile')

occurrence_per_city = get_occurence_of_word_per_city(word = search)
cities = cities.merge(occurrence_per_city, left_on='city', right_on='city')

st.plotly_chart(plot_map(size='word_count', color='word_count', range_color=None, color_continuous_scale=None))


if st.button('View Articles Including Searched Word'):
	st.write('Below you can find a list of articles that include the searched word or phrases exactly or as part of a word.')
	st.button('Collapse all News', key=2)
	
	for city in news.keys():
		rank = 1
		city_name_printed = False
		for headline in news[city]:

			if search.lower() in headline['title'].lower():
				
				if not city_name_printed:
					st.write(f'**{city}**')
					city_name_printed = True

				title = headline['title']
				link = headline['link']
				pubdate = headline['pubdate'][:-13]
				st.write(f'{rank}. {title} [[Link]]({link}) - Date: {pubdate}')
				rank += 1
	if rank == 1:
		st.write('There are no articles available that include the searched word.')
	st.button('Collapse all News', key=3)

st.subheader('Most Frequent Words per City')
st.write('This section provides a chart of the most frequent words per city you selected. Little hint: Try to search for words that appear in this bar char in the section above and have a look at this bar chart again!')
selected_city_word = st.selectbox(label='About which city do you want to know more?', options=cities['city'], key=1)
if len(news[selected_city_word]) > 0:
	st.plotly_chart(plot_bar(color_word = search))
else:
	st.write('There are unfortunately not enough news availabe for the selected city to plot a chart with the most frequent words. Please try again with a different city.')

st.subheader('Settings')

st.write('The news have been updated at {} UTC (WET-1)'.format(date_time))
st.write('Please press the button below to reload the news. The button needs to be pressed twice to reload.')
if st.button('Reload News'):
	st.write('If you really want to reload please press the button above again.')
	
	# legacy_caching will be removed in a future version of streamlit
	st.legacy_caching.caching.clear_cache()

st.write('If you want to change the country of the analysis, you can to so in the currently collapsed sidebar which can be displayed with a click of the arrow in the top left corner of this page. Please note that this program is not optimized for other countries than Ukraine. During this project we make use of the free version of the worldcities.csv file by simplemaps found under https://simplemaps.com/data/world-cities.')
