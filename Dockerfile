FROM jupyter/scipy-notebook:621b96ed75cb

ENV MKL_THREADING_LAYER GNU

RUN conda install -y -c conda-forge \
    'awscli=1.15.62' \
    'mkl=2018.0.3' \
    'mkl-service=1.1.2' \
    'pymc3=3.5' \
    'pyarrow=0.9.0'

RUN pip install sagemaker==1.7.1

ADD ./train /usr/local/bin/train
ADD ./train.py /usr/local/bin/train.py
USER root
