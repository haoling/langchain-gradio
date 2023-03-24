FROM fuyu-quant/gradio-docker
RUN mkdir /app
COPY requirements.txt /app
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
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
COPY --link --chown=1000 ./ /app
ENTRYPOINT ["python", "app.py"]

