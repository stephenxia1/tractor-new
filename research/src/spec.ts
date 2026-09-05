// Writes spec.json: the machine-readable mirror of the wire layout that
// py/tractor_rl/layout.py loads and asserts against. Run after any edit
// to record.ts:  npm run spec
import { writeFileSync } from 'node:fs';
import { layoutSpec } from './record';

const target = new URL('../spec.json', import.meta.url);
writeFileSync(target, `${JSON.stringify(layoutSpec(), null, 2)}\n`);
console.log(`wrote ${target.pathname}`);
