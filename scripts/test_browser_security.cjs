// Copyright (c) Microsoft. All rights reserved.

const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('docs/javascripts/charts.js', 'utf8');
const context = vm.createContext({});
vm.runInContext(source.slice(0, source.indexOf('(function () {')), context);
vm.runInContext(`
  const merged = deepMerge({}, JSON.parse('{"__proto__":{"polluted":true},"constructor":{"prototype":{"polluted":true}},"options":{"color":"blue"}}'));
  if (({}).polluted !== undefined || merged.options.color !== 'blue') throw new Error('unsafe merge');
  if (Object.hasOwn(merged, '__proto__') || Object.hasOwn(merged, 'constructor')) throw new Error('unsafe keys');
`, context);
const handlers = {};
let reads = 0;
const worker = vm.createContext({
  addEventListener: (name, handler) => { handlers[name] = handler; },
  self: {location: {origin: 'https://example.com'}, clients: {
    get: async () => { reads++; return {id: 'client'}; },
    matchAll: async () => [],
  }},
});
vm.runInContext(fs.readFileSync('dashboard/static/mockServiceWorker.js', 'utf8'), worker);
(async () => {
  await handlers.message({origin: 'https://attacker.invalid', source: {id: 'client'}, data: 'MOCK_ACTIVATE'});
  assert.equal(reads, 0);
  await handlers.message({origin: 'https://example.com', source: {id: 'client'}, data: 'UNKNOWN'});
  assert.equal(reads, 1);
  console.log('Chart pollution and worker origin regressions passed');
})();
