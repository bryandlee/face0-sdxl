FROM python:3.10

RUN apt-get update && apt-get install libgl1-mesa-glx libglib2.0-dev -y

RUN mkdir ./sdxl
RUN --mount=type=bind,source=./,target=./sdxl \
    pip install -r ./sdxl/requirements.txt --no-cache-dir

RUN apt-get install tmux zip unzip vim -y
RUN wget https://raw.githubusercontent.com/gpakosz/.tmux/master/.tmux.conf -O ~/.tmux.conf

CMD ["/bin/bash"]
