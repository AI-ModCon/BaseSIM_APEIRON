# Use the official Python image as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install poetry
RUN pip install --no-cache-dir poetry poetry-plugin-export

# Copy the dependency files to the working directory
COPY pyproject.toml poetry.lock ./

# Using poetry to make a requirements.txt then pip to do the install.
# Pip is aggressive with storage space.
RUN poetry export -f requirements.txt -o requirements.txt --without-hashes --with dev

RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

