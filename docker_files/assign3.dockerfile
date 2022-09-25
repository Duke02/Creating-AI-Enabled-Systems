FROM jhebeler/classtank:705.603.jupyterlab

RUN mkdir ./assign3
RUN mkdir ./assign3/data

COPY ./MayTrystan_Assign3.py ./assign3/main.py
COPY ./data/Assign3/Musical_instruments_reviews.csv ./assign3/data/music_reviews.csv

WORKDIR ./assign3
CMD python3 ./main.py

LABEL maintainer="Trystan May"
LABEL version="0.3"
LABEL description="Trystan May - Creating AI-Enabled Systems - Fall 2022 - Assignment 3 - Module 4"

EXPOSE 8787
EXPOSE 8888
EXPOSE 5000
