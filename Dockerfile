# docker build -t . 
FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN mkdir -p src
ADD code src/bert

CMD ["/bin/bash"]