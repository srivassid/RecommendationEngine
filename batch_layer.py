from imdb import IMDb
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gremlin_python.driver import client, serializer
import numpy as np
import random
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError
import time
from multiprocessing import Pool
import multiprocessing

pd.options.display.width = 0

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
random.seed(0)

def pre_process_plot_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

    text = text.lower()
    temp_sent = []
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES:
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)

        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)

    finalsent = ' '.join(temp_sent)
    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    return finalsent

def get_data(type):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")  # wd:Q11424
    sparql.setQuery("""
            SELECT DISTINCT ?item ?itemLabel WHERE {
            ?item """ + type + """
            ?item wdt:P577 ?pubdate.
            FILTER((?pubdate >= "2020-01-01T00:00:00Z"^^xsd:dateTime) && (?pubdate <= "2020-10-15T00:00:00Z"^^xsd:dateTime))
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                }
            """)
    sparql.setReturnFormat(JSON)
    counter = 0
    while (counter == 0):
        try:
            results = sparql.query().convert()
            counter = 1
        except HTTPError as e:
            print("Repeating")
        except Exception as e:
            print("Exception " + str(e) + " repeating ")
        if counter != 0:
            break

    df = pd.json_normalize(results['results']['bindings'])[['itemLabel.value']]
    if df.empty:
        return df[['itemLabel.value']]
    else:
        return df

def get_plot(df):
    df['plot'] = df['movie'].apply(lambda x: plot(x))
    return df

def plot(x):
    ia = IMDb()
    try:
        return ia.get_movie(ia.search_movie(x)[0].movieID)['plot'][0].split('::')[0] + \
               (ia.get_movie(ia.search_movie(x)[0].movieID)['title'])
    except Exception as e:
        return x

def create_df():
    movie = 'wdt:P31 wd:Q11424.'
    tv_shows = 'wdt:P31/wdt:P279* wd:Q5398426.'

    movie_df = get_data(movie)
    # print(movie_df)
    tv_df = get_data(tv_shows)
    # print(tv_df)
    df = pd.concat([movie_df,tv_df])
    df = df.rename(columns = {'itemLabel.value':'movie'})

    n_cores = multiprocessing.cpu_count()
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(get_plot, df_split))
    pool.close()
    pool.join()

    df['plot_processed'] = df['plot'].apply(lambda x:pre_process_plot_text(x))
    df = df.drop(columns=['plot'])
    df.to_csv('all_movies_tv.csv',index=False,sep=',')
    return df

def send_data(df):
    # df = pd.read_csv('movies/all_movies_tv.csv')
    df['movie'] = df['movie'].apply(lambda x: x.replace("'","") + "_" + str(random.randint(1,1000000)))
    # df = df[~df.movie.str.startswith('Q')]
    df = df.dropna(subset=['plot_processed'])
    df['name'] = df['movie']
    df = df.set_index('movie')
    tfidfvec = TfidfVectorizer()
    tfidf_movieid = tfidfvec.fit_transform((df["plot_processed"]))
    cos_sim = cosine_similarity(tfidf_movieid, tfidf_movieid)
    return df, cos_sim

def recommendations( df, cosine_sim):
    df_all = pd.DataFrame(columns={'name','plot_processed','rec_mov'})
    for title in (df['name'].tolist()):
        indices = pd.Series(df.index)
        recommended_movies = []
        df['rec_mov'] = np.empty((len(df), 0)).tolist()
        index = indices[indices == title].index[0]
        similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending=False)
        # top_10_movies = list(similarity_scores.iloc[1:11].index)
        for i,row in similarity_scores.iloc[1:11].iteritems():
            recommended_movies.append((list(df.index)[i], row))

        df.loc[title,['rec_mov']] = [recommended_movies]
        df_all = df_all.append({'id':title,'plot_processed':df['plot_processed'].loc[title],
                                'rec_mov':recommended_movies, 'name':title.split('_')[0]}, ignore_index=True)
    return df_all

def insert_graph(client, df):
    for i, row in df.iterrows():
        print(row['name'].replace("'",""), row['plot_processed'], row['rec_mov'])

        query = "g.addV('movie').property('id', '" + (row['id']) + \
                "').property('name', '" + str(row['name']) + \
                "').property('plot', '" + str(row['plot_processed']) + \
                "').property('pk', 'pk')"

        callback = client.submitAsync(query)
        if callback.result() is not None:
            try:
                print("\tInserted this vertex:\n\t{0}\n".format(
                callback.result().one()))
            except Exception as e:
                print("exception " + str(e))
        else:
            print("Something went wrong with this query: {0}".format(query))

def create_edges(client, df):
    for i, row in df.iterrows():
        for j in (row['rec_mov']):
            print(j[0],j[1])
            bound = "g.V('" + row['id'] + "').bothE().where(otherV().hasId('" + str(j[0]) + "'))"
            print('bound')
            print(bound)
            callback_bound = client.submitAsync(bound)
            edge = []

            for result in callback_bound.result():
                edge.append(result[0])

            print("Title " + row['id'] + " Recommendation " + j[0])

            if (len(edge) == 0) :
                try:
                    print("Empty in edge")
                    print("Will make a connection")
                    query = "g.V('" + row['id'] + "').addE('recommends').to(g.V('" + str(j[0]) + "')).property('weight'," + str(j[1]) + ")"
                    print('query is')
                    print(query)
                    callback = client.submitAsync(query)
                    if callback.result() is not None:
                        print("\tInserted this edge:\n\t{0}\n".format(
                            callback.result().one()))
                    else:
                        print("There was a problem with thr query\n\t{0}\n".format(query))
                except Exception as e:
                    print("exception " + str(e))
            else:
                print("Edge already exists")

if __name__ == '__main__':
    start = time.time()
    df = create_df()
    rec_df = (recommendations(df=send_data(df)[0], cosine_sim=send_data(df)[1]))

    clien = client.Client('<your-account>', 'g',
                                  username="<your-username>",
                                  password="<your-password>",
                                  message_serializer=serializer.GraphSONSerializersV2d0()
                                  )
    print(time.time() - start)
    insert_graph(client, rec_df)
    create_edges(client, rec_df)