# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code to the container
COPY . .

# Expose the app port
EXPOSE 8080

# Run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "main:app"]

