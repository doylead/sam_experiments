FROM python:3.9-slim-buster
WORKDIR /app
COPY layers/python/requirements.txt ./layers/python/requirements.txt
RUN pip install -r ./layers/python/requirements.txt
COPY . .
CMD ["flask", "run"]