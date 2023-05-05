
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ros1_distro=noetic
###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=user \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:iscilab" | chpasswd

## basic tools

RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    sudo \
    software-properties-common \
    git \
    cmake \
    wget \
    make \
    vim \
    build-essential \
    libblkid-dev \
    e2fslibs-dev \
    libboost-all-dev \
    libaudit-dev \
    tzdata \
    python3-pip

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN pip3 install --upgrade pip

## Dependency installation
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# RUN pip3 install open3d==v0.12.0 --ignore-installed PyYAML==3.13
# COPY ./requirement/package_apt.txt /
# COPY ./requirement/requirement_pip.txt /
# RUN apt update && xargs -a /package_apt.txt apt install -y
# RUN pip3 install -r /requirement_pip.txt
# RUN rm /package_apt.txt /requirement_pip.txt

## ROS1 installation
# RUN apt update && sudo apt install -y curl gnupg2 lsb-release
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
# RUN apt update
# RUN apt install -y ros-$ros1_distro-desktop-full
 
# # Install ROS 1 build tools
# RUN apt install -y python3-rosdep python3-rosinstall-generator python3-vcstool build-essential
# RUN apt-get install -y python3-pcl

# RUN echo "ROS $ros1_distro installed successfully"

## ROS2 installation
RUN locale

RUN sudo apt update && sudo apt install locales
RUN sudo locale-gen en_US en_US.UTF-8
RUN sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8

RUN locale

RUN sudo apt install software-properties-common -y
RUN sudo add-apt-repository universe

RUN sudo apt update && sudo apt install curl -y
RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN sudo apt update

RUN sudo apt install ros-foxy-desktop -y

RUN sudo apt install python3-colcon-common-extensions -y
RUN sudo apt install python3-colcon-mixin
RUN colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml
RUN colcon mixin update default
RUN sudo apt install python3-vcstool


## Setup ompl
# COPY ./requirement/setup_repo.sh /
# RUN sh /setup_repo.sh
# RUN rm /setup_repo.sh

## setting
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL) ALL" > /etc/sudoers
RUN echo "root ALL=(ALL) ALL" > /etc/sudoers

# GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# setup entrypoint
# COPY ./ros_bashrc /

# ENTRYPOINT ["/ros_bashrc"]

USER ${USER}
WORKDIR ${HOME}
RUN ls
