# Dockerfile
# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model (using the correct names)
COPY app.py .
COPY banana_nasik_rf_model_basic.pkl .  # <-- CORRECTED MODEL NAME
COPY banana_nasik_history.csv .         # <-- ADDED HISTORY FILE (Required by app.py)

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]