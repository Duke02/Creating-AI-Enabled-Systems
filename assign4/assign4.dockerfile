FROM jhebeler/classtank:705.603.jupyterlab

RUN mkdir ./assign4
RUN mkdir ./assign4/data

COPY ./assign4/main.py ./assign4/main.py
COPY ./data/Assign4/cars.csv ./assign4/data/cars.csv
COPY ./requirements.txt ./assign4/

RUN pip install -r ./assign4/requirements.txt

WORKDIR ./assign4
CMD python3 ./main.py

LABEL maintainer="Trystan May"
LABEL version="0.4"
LABEL description="Trystan May - Creating AI-Enabled Systems - Fall 2022 - Assignment 4 - Module 5"

EXPOSE 8787
EXPOSE 8888
EXPOSE 5000
