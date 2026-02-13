FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel

# sudo 설치
RUN apt-get update && apt-get install -y sudo vim git && rm -rf /var/lib/apt/lists/*

# 호스트와 동일한 UID/GID 설정 (빌드 시 override 가능)
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

# 그룹 및 유저 생성, sudo 권한 부여 (비밀번호 생략)
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# pip의 externally-managed-environment 제한 해제 (Docker 컨테이너이므로 안전)
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# Python 의존성 설치 (이미지에 bake하여 재현성 보장)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 접속 시 기본 사용자 및 경로 설정
USER $USERNAME
WORKDIR /home/$USERNAME

# conda 및 NVCC 경로 보장
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH

# NVIDIA 런타임 설정
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 멀티 GPU (NCCL) 최적화
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=eth0
