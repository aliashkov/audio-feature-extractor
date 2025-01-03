FROM node:20

WORKDIR /usr/src/app

# Install app dependencies
COPY package*.json ./
RUN npm install bullmq@5.10.3
RUN npm install

# Bundle app source
COPY . .

# Expose port
EXPOSE 3001

# Start command
CMD [ "npm", "start" ]
