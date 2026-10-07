# Use a lightweight, stable Python image
FROM python:3.11-slim

WORKDIR /app

# Copy dependency list and install them, when changes made only in python codes skips install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 8000

# Command that runs when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]