docker run -p 8000:8000 --restart=always --runtime=nvidia --gpus=all --env SUNO_USE_SMALL_MODELS=0 --env AUTH_TOKEN='' --detach tts_api
