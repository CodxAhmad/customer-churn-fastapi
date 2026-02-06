# Use Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing pyc files & stream logs immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy only requirements first (for Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app (replace 'main:app' if your app object or file differs)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
