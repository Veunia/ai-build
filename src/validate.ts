import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import Ajv, { DefinedError } from 'ajv';
import addFormats from 'ajv-formats';

import { Workflow } from './types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const schemaPath = path.resolve(__dirname, '../schemas/n8n-workflow.schema.json');
const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf-8'));

const ajv = new Ajv({
  allErrors: true,
  strict: true,
  allowUnionTypes: true,
});
addFormats(ajv);

const validator = ajv.compile<Workflow>(schema);

export function validateWorkflow(input: unknown): Workflow {
  if (!validator(input)) {
    const details = (validator.errors ?? [])
      .map((error: DefinedError) => `${error.instancePath || '/'} ${error.message}`)
      .join('\n');
    throw new Error(`Workflow invalide selon le schéma n8n:\n${details}`);
  }
  return input as Workflow;
}

function validateFile(filePath: string): void {
  const absolute = path.resolve(process.cwd(), filePath);
  const raw = fs.readFileSync(absolute, 'utf-8');
  const data = JSON.parse(raw);
  validateWorkflow(data);
  console.log(`✅ Workflow valide: ${absolute}`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const target = process.argv[2];
  if (!target) {
    console.error('Usage: pnpm val <chemin-vers-workflow.json>');
    process.exitCode = 1;
  } else {
    try {
      validateFile(target);
    } catch (error) {
      console.error('❌ Validation échouée.');
      if (error instanceof Error) {
        console.error(error.message);
      }
      process.exitCode = 1;
    }
  }
}
