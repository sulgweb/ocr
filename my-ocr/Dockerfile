FROM python:3.7

MAINTAINER xiaoyu "xiaoyu@sulg.top"

COPY ./requirements.txt /requirements.txt

WORKDIR /
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt update
RUN apt install libgl1-mesa-glx -y
RUN pip config set global.index-url https://mirror.baidu.com/pypi/simple
RUN pip install --upgrade pip
RUN pip install paddlepaddle==2.3.0
RUN pip install opencv-python-headless==4.4.0.46 opencv-contrib-python==4.4.0.46 paddleocr==1.1.1
RUN pip install Flask==2.1.2 Flask_RESTFul==0.3.9
RUN pip install -r requirements.txt
RUN pip install jieba
RUN pip install protobuf==3.20.1
RUN mkdir image

COPY . /


ENTRYPOINT [ "python" ]
EXPOSE 8866

CMD [ "app/app.py" ]