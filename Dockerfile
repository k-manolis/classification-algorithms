# user an official python runtime as a parent image
FROM python:3

#set the working directory to /app
WORKDIR /usr/src/app

COPY requirements.txt ./

#Install any needed packages specified in requirmentrs.txt
RUN pip install --no-cache-dir -r requirements.txt

ADD . /code

COPY . .

#Start the app
CMD ["python", "./main.py"]