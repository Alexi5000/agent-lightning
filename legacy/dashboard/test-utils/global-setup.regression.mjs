// Copyright (c) Microsoft. All rights reserved.

import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import net from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';

test('a server that becomes ready after ten seconds remains owned and alive', { timeout: 40000 }, async () => {
  const probe = net.createServer();
  await new Promise((resolve) => probe.listen(0, '127.0.0.1', resolve));
  const { port } = probe.address();
  await new Promise((resolve) => probe.close(resolve));
  process.env.VITEST_PYTHON_PORT = String(port);
  const directory = await mkdtemp(path.join(tmpdir(), 'lightning-startup-'));
  const runner = path.join(directory, 'runner.mjs');
  const fixture = new URL('./delayed-python-server.mjs', import.meta.url).href;
  await writeFile(runner, `#!/usr/bin/env node\nimport(${JSON.stringify(fixture)});\n`, { mode: 0o700 });
  process.env.VITEST_PYTHON_BIN = runner;
  delete process.env.VITEST_PYTHON_URL;
  const { default: setup } = await import('../vitest.global-setup.mjs');
  const teardown = await setup();
  const health = `http://127.0.0.1:${port}/v1/agl/health`;
  try {
    await sleep(300);
    assert.equal((await fetch(health)).status, 200);
  } finally {
    await teardown();
    await rm(directory, { recursive: true, force: true });
  }
  await assert.rejects(fetch(health));
});
