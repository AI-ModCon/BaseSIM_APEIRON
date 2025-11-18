# Use the official Python image as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy the dependency files to the working directory
# Copy the rest of the application code to the working directory
COPY . .
#COPY pyproject.toml poetry.lock README.md ./


# Install project dependencies
RUN poetry install



# No CMD or ENTRYPOINT is specified as this is for CI/CD
