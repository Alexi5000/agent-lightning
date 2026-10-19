// Copyright (c) Microsoft. All rights reserved.

import assert from 'node:assert/strict';
import net from 'node:net';
import { test } from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

test('a server that becomes ready after ten seconds remains owned and alive', { timeout: 40000 }, async () => {
  const probe = net.createServer();
  await new Promise((resolve) => probe.listen(0, '127.0.0.1', resolve));
  const { port } = probe.address();
  await new Promise((resolve) => probe.close(resolve));
  process.env.VITEST_PYTHON_PORT = String(port);
  process.env.VITEST_PYTHON_BIN = fileURLToPath(new URL('./delayed-python-server.mjs', import.meta.url));
  delete process.env.VITEST_PYTHON_URL;
  const { default: setup } = await import('../vitest.global-setup.mjs');
  const teardown = await setup();
  const health = `http://127.0.0.1:${port}/v1/agl/health`;
  try {
    await sleep(300);
    assert.equal((await fetch(health)).status, 200);
  } finally {
    await teardown();
  }
  await assert.rejects(fetch(health));
});
