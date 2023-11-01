# FROM ubuntu:23.10
FROM python:3.9.17
WORKDIR /digits
COPY . /digits/
VOLUME test_vol4
RUN pip3 install --no-cache-dir -r /digits/requirements.txt
RUN mkdir -p models
CMD ["pytest"]
CMD ["python3","/digits/exp.py"]