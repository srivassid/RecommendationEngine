from gremlin_python.driver import client, serializer

def get_recommendation(cosmos_client, movie):
    query = "g.V().has('name','" + movie + "').bothE()"
    callback = cosmos_client.submitAsync(query)
    for result in callback.result():
        for i in result:
            if i['inV'].startswith(movie):
                pass
            else:
                print(i['inV'].split('_')[0])

            if i['outV'].startswith(movie):
                pass
            else:
                print(i['outV'].split('_')[0])

if __name__ == '__main__':
    cosmos_client = client.Client('wss://account-cosmos-db.gremlin.cosmosdb.azure.com:443/', 'g',
                           username="/dbs/GraphDB1/colls/MovieGraph3",
                           password="np4uDpHJIdpY1JCWHJpLX1QRxFYxVVT4mnE55qIG3MmrFDoqKHmG7Spptp7dxx8LFLD5D6xeOsRjO1YMpyjZtA==",
                           message_serializer=serializer.GraphSONSerializersV2d0()
                           )
    movie = 'Sister'
    get_recommendation(cosmos_client, movie)

