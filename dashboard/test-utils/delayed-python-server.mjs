// Copyright (c) Microsoft. All rights reserved.

import http from 'node:http';

setTimeout(() => {
  const server = http.createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end('{"status":"ok"}');
  });
  server.listen(Number(process.env.VITEST_PYTHON_PORT), '127.0.0.1');
}, 11000);
