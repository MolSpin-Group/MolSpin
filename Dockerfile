FROM debian:13 AS build-env

RUN ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y libblas-dev libboost-dev libopenblas-dev cmake gcc make build-essential curl

WORKDIR /tmp/
RUN curl -X GET -L -o libarmadillo.tar.xz https://sourceforge.net/projects/arma/files/armadillo-12.8.4.tar.xz/download && \
    tar -xJf /tmp/libarmadillo.tar.xz && \
    cd armadillo-12.8.4 && \
    mkdir build && \
    cd build && \
    cmake -D OPENBLAS_PROVIDES_LAPACK=true .. && \
    make -j 2 && \
    make install 

WORKDIR /root

FROM build-env 

COPY . .

RUN make -j 2

RUN cp molspin /usr/bin

RUN rm -rf root/*

ENTRYPOINT ["molspin"]