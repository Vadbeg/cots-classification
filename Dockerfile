FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

#COPY . /app
WORKDIR /app
ADD ./requirements-docker.txt /app/requirements-docker.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip && pip3 install -r requirements-docker.txt && \
    pip3 install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip

#ADD . /app
COPY . /app

ENTRYPOINT ["/bin/bash"]
