# Makefile for managing Docker container operations

# Define variables
IMAGE_NAME := proteinclassifier-inference
CONTAINER_NAME := proteinclassifier-dev
VERSION ?= latest

# Default target executed when no arguments are given to make
all: build run

# Build the Docker image
build:
	@echo -n "Enter a version tag [latest]: "; \
	read TAG; \
	TAG=$${TAG:-latest}; \
	docker build -t $(IMAGE_NAME):$$TAG -t $(IMAGE_NAME):latest .

# Run the Docker container
run:
	docker run --name $(CONTAINER_NAME) -d -p 80:80 $(IMAGE_NAME):latest

# Stop and remove the Docker container
stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

# Remove the Docker image
clean-image:
	docker rmi $(IMAGE_NAME)

# Clean up stopped containers and dangling images
clean-docker:
	docker system prune -f

# Help command to list available commands
help:
	@echo "Available commands:"
	@echo "  all          - Build the Docker image (default)"
	@echo "  build        - Build the Docker image"
	@echo "  run          - Run the Docker container"
	@echo "  stop         - Stop and remove the Docker container"
	@echo "  clean-image  - Remove the Docker image"
	@echo "  clean-docker - Clean up stopped containers and dangling images"
	@echo "  help         - Display this help message"

# Declare phony targets
.PHONY: all build run stop clean-image clean-docker help
