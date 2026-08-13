FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
RUN pip install --no-cache-dir .
EXPOSE 4000
CMD ["uvicorn", "rooom_litellm.app:app", "--host", "0.0.0.0", "--port", "4000"]
