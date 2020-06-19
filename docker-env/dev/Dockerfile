FROM ubuntu:18.04

# Base dependencies and utilities

RUN apt update && \
    apt upgrade -y
    
RUN apt install -y gdebi
RUN apt install -y wget

RUN wget http://security.ubuntu.com/ubuntu/pool/main/libx/libxfont/libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb
RUN wget http://security.ubuntu.com/ubuntu/pool/universe/x/xorg-server/xvfb_1.18.4-0ubuntu0.7_amd64.deb

RUN gdebi -n libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb
RUN gdebi -n xvfb_1.18.4-0ubuntu0.7_amd64.deb

USER root

ADD launch.sh /root/launch
RUN chmod +x /root/launch

ADD build /root/lve
RUN chmod +x /root/lve/lve.x86_64
