#FROM python:3.9.17
FROM tensorflow/tensorflow:devel-gpu
#FROM tensorflow/tensorflow:2.10.0-gpu
#FROM python:3.8-slim

WORKDIR /link-analyzer

#Install Python 3.9 and pip
#RUN apt-get update && apt-get install -y python3.9 python3.9-distutils python3.9-dev python3-pip && \
#    python3.9 -m pip install --upgrade pip && \
#    ln -s /usr/bin/python3.9 /usr/bin/python

# Copy the entire project (including the .keras file) into the container
COPY . /link-analyzer

#RUN pip install psycopg2 python-dotenv pytz numpy matplotlib statsmodels flask tensorflow scikit-learn pandas
#RUN pip install psycopg2-binary python-dotenv pytz numpy matplotlib statsmodels flask tensorflow scikit-learn pandas
#RUN pip install --no-cache-dir numpy==1.21.0 pandas==1.3.0 psycopg2-binary python-dotenv pytz matplotlib statsmodels flask tensorflow scikit-learn
#RUN pip install --no-cache-dir tensorflow numpy pandas psycopg2-binary python-dotenv pytz matplotlib statsmodels flask scikit-learn
RUN pip install --no-cache-dir numpy>=1.20.3 pandas psycopg2-binary python-dotenv pytz matplotlib statsmodels flask tensorflow scikit-learn

    