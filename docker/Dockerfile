# FROM ubuntu:23.10
FROM python:3.9.17
WORKDIR /digits
COPY . /digits/
RUN pip3 install --no-cache-dir -r /digits/requirements.txt
RUN mkdir -p models
# CMD ["pytest"]
ENV FLASK_APP=api/app
# CMD ["flask","run","--host=0.0.0.0"]
CMD ["python","api/app.py"]

# CMD ["python3","/digits/exp.py"]
