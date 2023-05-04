# 1
FROM python:3.9

# 2
RUN pip install Flask gunicorn
RUN pip install pandas
RUN pip install numpy
RUN pip install -U scikit-learn

# 3
COPY src/ /app
WORKDIR /app

# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app