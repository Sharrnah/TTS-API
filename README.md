# Installation
- Make sure you have Docker installed. preferably with CUDA support.
- Clone the Bark Repository into the cache/ folder with lfs support.
  ```sh
  git lfs install
  git clone https://huggingface.co/suno/bark ./cache/suno/bark_v0/
  ```

- Build the Docker image using
  `docker build . -t tts_api`
- Run the Docker image with
  `docker run -p 8000:8000 --restart=always --runtime=nvidia --gpus=all --env SUNO_USE_SMALL_MODELS=0 --env AUTH_TOKEN='' --detach tts_api`

- Access the Swagger API Docs on http://localhost:8000/docs (or any other Port if changed in docker run command)
