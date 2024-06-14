# Set base image (host OS)
FROM python:3.9

# By default, listen on port 5000
EXPOSE 5000

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Copy subdirectories to the working directory
COPY theROCK ./theROCK
COPY templates ./templates
COPY static ./static

# Copy the content of the local src directory to the working directory
COPY application.py .

# Copy the model to the working directory
COPY model.py .

# Install any dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Set the PYTHONPATH environment variable to include the folder1 directory
ENV PYTHONPATH "${PYTHONPATH}:/app/theROCK/models/rnn"
ENV PYTHONPATH "${PYTHONPATH}:/app/theROCK/templates"

# Specify the command to run on container start
CMD [ "python", "./application.py" ]