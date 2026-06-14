FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AML_PROFILE=dev

COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
