docker build -t movie_recommend .



docker tag movie_recommend siddockerregistry.azurecr.io/movie_recommend



docker push siddockerregistry.azurecr.io/movie_recommend
