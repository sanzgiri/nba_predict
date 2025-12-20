# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER
USER root

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Install Tensorflow
#RUN conda install --quiet --yes \
#    'tensorflow=1.11*' \
#    'keras=2.2*' && \
#    conda clean -tipsy && \
#    fix-permissions $CONDA_DIR && \
#    fix-permissions /home/$NB_USER

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    rm -rf /var/lib/apt/lists/* && \
    sudo apt-get update && \ 
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL default-jre && \
    $PIP_INSTALL lxml html5lib geopy catboost \
                 request tabulate "colorama>=0.3.8" category_encoders streamlit \
                 statsmodels basketball_reference_web_scraper unidecode
    #pip uninstall -y h2o && \
    #pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

#EXPOSE 8888 54321
