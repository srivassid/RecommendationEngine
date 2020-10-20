import datetime
import uuid
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError
from msrest.authentication import TopicCredentials
from azure.eventgrid import EventGridClient
from azure.eventgrid.models import EventGridEvent
import pandas as pd
import logging, time
from datetime import date, timedelta

TOPIC_ENDPOINT = "movietopic1.westeurope-1.eventgrid.azure.net"

EVENT_GRID_KEY = '0rX2S0JW6EhZ0pSYRGi1ZeDbn7baypPpMF0BYZg4of4='

def get_data(type):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    today = date.today()
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)
    sparql.setQuery("""
                SELECT DISTINCT ?item ?itemLabel WHERE {
                ?item """ + type + """
                ?item wdt:P577 ?pubdate.
                FILTER((?pubdate >= '""" + str(day_before) + """'^^xsd:dateTime) && (?pubdate <= '""" + str(yesterday) + """'^^xsd:dateTime))
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
    print(results)
    df = pd.json_normalize(results['results']['bindings'])[['itemLabel.value']]
    df = df.rename(columns={'itemLabel.value':'movie'})
    print(df)
    return df

def build_events_list(df):
    result = []
    for i, row in df.iterrows():
        result.append(EventGridEvent(
            id = uuid.uuid4(),
            subject='Row 1',
            data = {
                'movie':row['movie'],
            },
            event_type='PersonalEventType',
            event_time=datetime.datetime.now(),
            data_version=2.0
        ))
    print('result is')
    print(result)
    return result

def run_example(df):

    credentials = TopicCredentials(
        EVENT_GRID_KEY
    )
    event_grid_client = EventGridClient(credentials)
    event_grid_client.publish_events(
        TOPIC_ENDPOINT,
        events=build_events_list(df)
    )
    print("Published events to Event Grid.")
    for i, row in df.iterrows():
        logging.info('Movie event has been sent: %s',row['movie'])

if __name__ == "__main__":
    while True:
        tv_shows = 'wdt:P31/wdt:P279* wd:Q5398426.'
        movie = 'wdt:P31 wd:Q11424.'
        df_tv = get_data(tv_shows)
        df_movie = get_data(movie)
        df = pd.concat([df_tv,df_movie])
        logging.info("new movies are")
        print(df)
        if df.empty is False:
            run_example(df)
        else:
            print("Empty df, will try again tomorrow")
        time.sleep(86400)
