# Dockerized LDAitkmitl
> Work as a part of Queueing System

**This version of code is meant to be run inside Docker environment with other services by using _docker-compose_**

## How to Run In Headless Mode
In case of debugging without other services. Follow these steps below

1. Make sure that you've Docker installed on your machine. You can [download it here](https://www.docker.com).
2. Build an image from Dockerfile. This is a one-time step. (This will take a while)
```shell script
docker build -t status_code .
```
3. You can config the request body to send by editing [json_request.json](json_request.json).
4. 