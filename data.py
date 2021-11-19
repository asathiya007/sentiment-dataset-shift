# import modules 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from texthero import preprocessing, clean


def _default_vectorize_text(data, vector_size=100, epochs=20):
    # clean text 
    cleaning_pipeline = [
        preprocessing.drop_no_content,
        preprocessing.lowercase,
        preprocessing.remove_whitespace,
        preprocessing.remove_diacritics,
        preprocessing.remove_brackets,
        preprocessing.remove_stopwords, 
        preprocessing.replace_punctuation
    ]
    data['vectorized_text'] = clean(data['text'], cleaning_pipeline)

    # train Doc2Vec model
    tagged_text = []
    for i in range(len(data)):
        tokens = data.loc[i, 'vectorized_text'].split(' ')
        tagged_text.append(TaggedDocument(tokens, [i]))
    model = Doc2Vec(vector_size=vector_size, min_count=1, epochs=epochs)
    model.build_vocab(tagged_text)
    model.train(tagged_text, total_examples=model.corpus_count, 
        epochs=epochs)

    # vectorize text 
    vectorized_text = []
    for i in range(len(data)):
        tokens = data.loc[i, 'vectorized_text'].split(' ')
        vectorized_text.append(list(model.infer_vector(tokens)))
    data['vectorized_text'] = vectorized_text

    # return data with vectorized text
    return data 

def default_posneg_loader(): 
    # load IMDB data 
    imdb_data = pd.read_csv('./datasets/IMDB/IMDBDataset.csv')
    imdb_data['dataset'] = 'IMDB'
    imdb_data['text'] = imdb_data['review']
    imdb_data = imdb_data[['text', 'sentiment', 'dataset']]

    # load movie reviews data 
    movie_reviews_data = pd.read_csv('./datasets/MovieReviews/movie_review.csv')
    movie_reviews_data['sentiment'] = movie_reviews_data['tag']
    movie_reviews_data['dataset'] = 'Movie Reviews'
    movie_reviews_data = movie_reviews_data[['text', 'sentiment', 'dataset']]

    # load financial news data 
    financial_news_data = pd.read_csv('./datasets/FinancialNews/all-data.csv', 
        encoding="ISO-8859-1", header=None)
    financial_news_data['text'] = financial_news_data[1]
    financial_news_data['sentiment'] = financial_news_data[0]
    financial_news_data['dataset'] = 'Financial News'
    financial_news_data = financial_news_data[['text', 'sentiment', 'dataset']]

    # concatenate datasets 
    posneg_data = pd.concat([imdb_data, movie_reviews_data, financial_news_data], 
        axis=0, ignore_index=True)

    # vectorize text 
    posneg_data = _default_vectorize_text(posneg_data)

    # return loaded datasets 
    return posneg_data 
    
def default_emotions_loader(): 
    # load emotions dataset 
    emotions_data = []
    file = open('./datasets/EmotionsDataset/train.txt')
    line = file.readline()[:-1]
    while line: 
        chunks = line.split(';')
        emotions_data.append(chunks)
        line = file.readline()[:-1]
    emotions_data = pd.DataFrame(emotions_data, columns=['text', 'sentiment'])
    emotions_data['dataset'] = 'Emotions Dataset'

    # load Twitter dataset 
    twitter_data = pd.read_csv('./datasets/Twitter/data.csv')
    twitter_data['text'] = twitter_data['Tweets']
    twitter_data['sentiment'] = twitter_data['Feeling']
    twitter_data['dataset'] = 'Twitter'
    twitter_data = twitter_data[['text', 'sentiment', 'dataset']]

    # load Crowd Flower dataset 
    crowdflower_data = pd.read_csv('./datasets/CrowdFlower/text_emotion.csv')
    crowdflower_data['text'] = crowdflower_data['content']
    crowdflower_data['dataset'] = 'Crowd Flower'
    crowdflower_data = crowdflower_data[['text', 'sentiment', 'dataset']]

    # concatenate datasets
    emotions_data = pd.concat([emotions_data, twitter_data, crowdflower_data], 
        axis=0, ignore_index=True)

    # vectorize text 
    emotions_data = _default_vectorize_text(emotions_data)
    
    # return emotions dataset 
    return emotions_data
