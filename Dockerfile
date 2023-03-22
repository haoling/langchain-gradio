FROM fuyu-quant/gradio-docker
COPY requirements.txt /root
RUN sed -i 's http://deb.debian.org http://cdn-aws.deb.debian.org g' /etc/apt/sources.list \
    && sed -i 's http://archive.ubuntu.com http://us-east-1.ec2.archive.ubuntu.com g' /etc/apt/sources.list \
    && sed -i '/security/d' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y \
        git \
        git-lfs \
        ffmpeg \
        libsm6 \
        libxext6 \
        cmake \
        libgl1-mesa-glx \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /root
COPY --link --chown=1000 ./ /root
ENTRYPOINT ["python", "app.py"]

