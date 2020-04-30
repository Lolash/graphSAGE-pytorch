FROM local-torch-geometric
WORKDIR /app
COPY . .
RUN ls
RUN pip install -r requirements.txt
CMD ["python3"]
