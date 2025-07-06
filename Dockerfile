# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
#COPY .env .env


# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "ytagent_backend:app", "--host", "0.0.0.0", "--port", "8000"]
