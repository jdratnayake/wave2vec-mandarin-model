FROM python:3.10

RUN apt-get update && \
    apt-get install -y libsndfile1


WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./app.py" ]