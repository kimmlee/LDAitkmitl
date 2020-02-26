FROM python:3.6
ENV PYTHONUNBUFFERED 1

# Install Java First Because this will not change (Because some libs need it)
RUN set -e; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ghostscript \
    ; \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 0xB1998361219BD9C9; \
    apt-add-repository 'deb http://repos.azulsystems.com/debian stable main'; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        zulu-8 \
    ; \
    apt-get clean; \
    rm -rf /var/tmp/* /tmp/* /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-service.txt /app/
RUN pip install -r requirements.txt --no-cache-dir && pip install -r requirements-service.txt --no-cache-dir
COPY . .
CMD ["python", "service.py"]