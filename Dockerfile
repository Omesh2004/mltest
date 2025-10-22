FROM pathwaycom/pathway:latest

# Set working directory
WORKDIR /app




# Copy the rest of the application code
COPY  app.py .

# Command to run the Pathway script
CMD [ "python", "./app.py" ]
