FROM python:3.11-slim
LABEL maintainer="Sirish Shrestha <sirish_dot_shrestha_at_gmail_dot_com"

WORKDIR /usr/src/app
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y gcc python3-dev && \
    pip install --upgrade pip && \
    apt-get purge && \
    apt-get clean all && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt .
COPY . /usr/src/app

RUN pip install -r requirements.txt && \
    pip install --no-cache-dir jupyterlab


EXPOSE 8888

CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]