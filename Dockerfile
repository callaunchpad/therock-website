# Set base image (host OS)
FROM python:3.9

# By default, listen on port 5000
EXPOSE 5000/tcp

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Copy subdirectories to the working directory
COPY theROCK ./theROCK

# Install any dependencies
RUN pip install -r requirements.txt

# Set the PYTHONPATH environment variable to include the folder1 directory
ENV PYTHONPATH "${PYTHONPATH}:/app/theROCK/models/rnn"

# Copy the content of the local src directory to the working directory
COPY app.py .

# Copy the model to the working directory
COPY model.py .

# Specify the command to run on container start
CMD [ "python", "./app.py" ]