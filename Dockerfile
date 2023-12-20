FROM python:3.10-slim
WORKDIR /app
RUN pip install pipenv
COPY ["Pipfile","Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["models/", "./models/"]
COPY ["predict.py", "./"]
CMD [ "gunicorn", "--bind=0.0.0.0:4041", "predict:app" ]
