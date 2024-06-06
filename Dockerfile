# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80 (if needed for a web application; otherwise, this can be removed)
EXPOSE 80

# Define environment variable (remove if not needed)
ENV NAME World

# Run the application
CMD ["python", "src/predict.py"]
