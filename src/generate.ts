import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import OpenAI from 'openai';

import { GenerateOptions, Workflow } from './types.js';
import { validateWorkflow } from './validate.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function loadFile(relativePath: string): Promise<string> {
  const absolute = path.resolve(__dirname, '..', relativePath);
  return fs.readFile(absolute, 'utf-8');
}

function extractJsonPayload(text: string): string {
  const fenced = text.match(/```json\s*([\s\S]+?)```/i);
  if (fenced?.[1]) {
    return fenced[1].trim();
  }
  const genericFence = text.match(/```\s*([\s\S]+?)```/);
  if (genericFence?.[1]) {
    return genericFence[1].trim();
  }
  return text.trim();
}

async function callOpenAI(options: GenerateOptions): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY manquant dans les variables d\'environnement.');
  }

  const client = new OpenAI({ apiKey });

  const [systemPrompt, devPrompt, fewShots, schema] = await Promise.all([
    loadFile('system/SYSTEM_PROMPT.fr.md'),
    loadFile('system/DEV_PROMPT.fr.md'),
    loadFile('system/FEW_SHOTS.md'),
    loadFile('schemas/n8n-workflow.schema.json'),
  ]);

  const response = await client.responses.create({
    model: options.model ?? 'gpt-4.1-mini',
    temperature: options.temperature ?? 0.2,
    input: [
      { role: 'system', content: systemPrompt },
      { role: 'developer', content: devPrompt },
      {
        role: 'system',
        content: `Schéma JSON strict (utiliser uniquement les propriétés définies) :\n${schema}`,
      },
      {
        role: 'user',
        content: `Exemples de workflows valides (few-shots) :\n${fewShots}`,
      },
      {
        role: 'user',
        content: `Spécification à implémenter :\n${options.spec}\n\nRetourne UNIQUEMENT le JSON du workflow.`,
      },
    ],
  });

  const outputText = (response.output_text ?? '').trim();
  if (!outputText) {
    throw new Error('Réponse vide reçue du modèle OpenAI.');
  }

  return extractJsonPayload(outputText);
}

export async function generateWorkflow(options: GenerateOptions): Promise<Workflow> {
  const jsonText = await callOpenAI(options);
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch (error) {
    throw new Error(`La réponse du modèle n'est pas un JSON valide.\n${jsonText}`);
  }

  return validateWorkflow(parsed);
}

async function writeWorkflowFile(workflow: Workflow, outputPath: string): Promise<void> {
  const content = `${JSON.stringify(workflow, null, 2)}\n`;
  await fs.writeFile(outputPath, content, 'utf-8');
}

function parseArgs(argv: string[]): GenerateOptions {
  const args = [...argv];
  const specParts: string[] = [];
  let outputPath: string | undefined;
  let model: string | undefined;
  let temperature: number | undefined;

  while (args.length > 0) {
    const token = args.shift();
    if (!token) continue;
    if (token.startsWith('--out=')) {
      outputPath = token.split('=').slice(1).join('=');
    } else if (token === '--out') {
      outputPath = args.shift();
    } else if (token.startsWith('--model=')) {
      model = token.split('=').slice(1).join('=');
    } else if (token === '--model') {
      model = args.shift();
    } else if (token.startsWith('--temperature=')) {
      const value = token.split('=').slice(1).join('=');
      temperature = Number(value);
    } else if (token === '--temperature') {
      const value = args.shift();
      if (value !== undefined) {
        temperature = Number(value);
      }
    } else {
      specParts.push(token);
    }
  }

  const spec = specParts.join(' ').trim();
  if (!spec) {
    throw new Error('Veuillez fournir une spécification en argument, par exemple: pnpm gen "Spec détaillée"');
  }

  return {
    spec,
    outputPath,
    model,
    temperature,
  };
}

async function runCli(): Promise<void> {
  try {
    const options = parseArgs(process.argv.slice(2));
    const outputPath = path.resolve(process.cwd(), options.outputPath ?? 'workflow.generated.json');
    const workflow = await generateWorkflow(options);
    await writeWorkflowFile(workflow, outputPath);
    console.log(`✅ Workflow généré et validé: ${outputPath}`);
  } catch (error) {
    console.error('❌ Échec de la génération du workflow.');
    if (error instanceof Error) {
      console.error(error.message);
    }
    process.exitCode = 1;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  runCli();
}
