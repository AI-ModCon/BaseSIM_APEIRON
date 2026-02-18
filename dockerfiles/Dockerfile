# Use the official Python image as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency files to the working directory
COPY pyproject.toml poetry.lock ./

# Install poetry, export requirements, install deps with CPU-only PyTorch,
# then clean up poetry -- all in one layer to minimize image size.
RUN pip install --no-cache-dir poetry poetry-plugin-export \
    && poetry export -f requirements.txt -o requirements.txt --without-hashes --with dev \
    && pip install --no-cache-dir -r requirements.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip uninstall -y poetry poetry-plugin-export \
    && rm requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

