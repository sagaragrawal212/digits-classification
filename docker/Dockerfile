# FROM ubuntu:23.10
FROM python:3.9.17
WORKDIR /digits
COPY . /digits/
RUN pip3 install --no-cache-dir -r /digits/requirements.txt
RUN mkdir -p models
CMD ["pytest"]
# RUN export FLASK_APP = app.py
# CMD["flask","run"]

# CMD ["python3","/digits/exp.py"]
