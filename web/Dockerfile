FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src ./src
COPY stream_test.html ./stream_test.html

RUN npm run build

FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY --from=builder /app/dist ./dist
COPY stream_test.html ./stream_test.html

ENV PORT=8080
EXPOSE 8080

CMD ["node", "dist/server.js"]
