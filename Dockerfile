FROM node:22-alpine AS frontend-build

WORKDIR /workspace/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py main.py model.py efficient_kan.py ./
COPY final_best_model.pth ./
COPY dataset/train ./dataset/train
COPY --from=frontend-build /workspace/web/dist ./web/dist

EXPOSE 10000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
