# Hugging Face Space (Docker SDK) for the Material Match web app.
# Built from the folder produced by:  python deploy/build_hf_space.py
FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    MATERIAL_MATCH_HOSTED=1
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small; other versions match the ones the model was trained with.
COPY --chown=user requirements-web.txt .
RUN pip install --no-cache-dir --user torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-web.txt

# Bake the MiniLM encoder into the image so start-up never depends on a download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
