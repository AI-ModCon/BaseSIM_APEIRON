# Use the official Python image as a parent image
FROM python:3.13-slim

# uv, pinned to the same minor series as the uv_build backend in pyproject.toml
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Use the image's Python instead of downloading one, and install without a
# cache (copying rather than hardlinking) to keep the image small.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# Install dependencies first, so this layer is reused when only the source
# changes. --locked fails the build if uv.lock is out of date.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Copy the rest of the application code and install the project itself
COPY . .
RUN uv sync --locked

# Put the project environment on PATH so `docker run image pytest` works
ENV PATH="/app/.venv/bin:$PATH"
