FROM node:16

# Install dependencies for WASM support
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3

WORKDIR /usr/src/app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source
COPY . .

EXPOSE 3000

CMD [ "node", "--experimental-wasm-threads", "app.js" ]
