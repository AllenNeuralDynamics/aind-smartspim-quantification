FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

# Necessary for vtkplotter
RUN apt-get update && apt-get install libgl1 -y
RUN apt-get install libxrender1

# vtkplotter works properly on python 3.9
RUN conda create -n quantification python=3.9 -y
SHELL ["conda", "run", "-n", "quantification", "/bin/bash", "-c"]

# aind-ng-link==1.0.6 \
RUN pip install -U --no-cache-dir \
    numpy==1.24.0 \
    dask[distributed]==2022.11.1 \
    zarr==2.13.6 \
    scikit-learn==1.5.1 \
    imlib==0.1.3 \
    tqdm==4.65.0 \
    antspyx \
    argschema==3.0.4 \
    ray==2.9.0 \
    pims==0.6.1 \
    xmltodict==0.13.0 \
    neuroglancer==2.36 \
    boto3==1.26.112 \
    vedo==2023.4.6 \
    ome-zarr==0.8.2 \
    aind-data-schema==1.0.0