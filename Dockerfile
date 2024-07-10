# Use an official Python runtime as the base image
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . .

# Ensure the cache directory has the correct permissions
RUN mkdir -p /app/cache && chown -R 1000:1000 /app/cache

# Ensure the lmss directory and files have the correct permissions
RUN mkdir -p /app/app/lmss && chown -R 1000:1000 /app/app/lmss

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME=LMSS_Entity_Recognizer

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]