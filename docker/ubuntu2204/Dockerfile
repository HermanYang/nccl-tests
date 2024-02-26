FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnuma-dev \
        numactl \
        python3 \
        python3-pip \
        && \
    rm -rf /var/lib/apt/lists/*


# Install Python dependencies   
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Remove the libnccl-dev and libnccl2 packages
RUN apt-get purge libnccl-dev libnccl2  -y --allow-change-held-packages

# Set the working directory
WORKDIR /nccl-tests