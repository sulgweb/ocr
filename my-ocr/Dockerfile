FROM python:3.7

MAINTAINER xiaoyu "xiaoyu@sulg.top"

COPY ./requirements.txt /requirements.txt

WORKDIR /
RUN pip config set global.index-url https://mirror.baidu.com/pypi/simple
RUN pip install --upgrade pip
RUN pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple
RUN pip install  opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64 paddleocr --no-cache-dir
RUN pip install Flask==2.1.2 Flask_RESTFul==0.3.9

COPY . /
RUN pip show paddlepaddle


ENTRYPOINT [ "python3" ]
EXPOSE 8866

CMD [ "app/app.py" ]