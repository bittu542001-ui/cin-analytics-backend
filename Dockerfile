# Dockerfile - Python + R combined for Render or any Docker host
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# 1) Install system packages, R, python, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common dirmngr gnupg apt-transport-https ca-certificates \
    build-essential libcurl4-openssl-dev libssl-dev libxml2-dev libpq-dev \
    wget curl git locales \
    python3 python3-venv python3-pip python3-dev python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# 2) Install R (CRAN)
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys '51716619E084DAB9' || true
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt-get update && apt-get install -y --no-install-recommends r-base r-base-dev

# 3) Ensure pip is up-to-date
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy files
COPY . /app

# 4) Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 5) Install required R packages used by our scripts
#    (this installs quietly; adjust package list if you need more)
RUN Rscript -e "install.packages(c('jsonlite','pROC','ResourceSelection','mice','DescTools'), repos='https://cloud.r-project.org')"

# 6) Expose port and start uvicorn
ENV PORT=8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
