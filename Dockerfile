# Step 1: Use an official Python base image
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Install dependencies
# Copy the requirements.txt to the container and install the dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the rest of the application files to the container
COPY backend/ ./

# Step 5: Expose the port the app runs on
EXPOSE 8000

# Step 6: Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
