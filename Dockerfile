FROM daskdev/dask:latest

RUN mkdir -p /workdir/btb && \
    apt-get update && \
    apt-get install -y build-essential

COPY setup.py MANIFEST.in /workdir/btb/
COPY btb /workdir/btb/btb
COPY benchmark /workdir/btb_benchmark
COPY benchmark/notebooks /workdir/notebooks
RUN pip install -e /workdir/btb[dev] -e /workdir/btb_benchmark

WORKDIR /workdir/notebooks
CMD jupyter notebook --ip 0.0.0.0 --allow-root
