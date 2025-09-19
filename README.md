# Générateur de workflows n8n

Cette application Node.js/TypeScript génère et valide des workflows [n8n](https://n8n.io) à partir d'une spécification textuelle, image ou vidéo.

## Prérequis

- [Node.js](https://nodejs.org/) >= 18
- [pnpm](https://pnpm.io/) >= 8
- Une clé API OpenAI disponible dans la variable d'environnement `OPENAI_API_KEY`
- Toutes les informations d'identification requises exposées via des variables `CREDENTIALS.*`

## Installation

```bash
pnpm install
```

## Génération d'un workflow

```bash
pnpm gen "Ma spec ultra-précise"
```

Le script construit une requête OpenAI à partir des prompts systèmes et développeur, applique les few-shots puis enregistre le JSON produit dans `workflow.generated.json`.

## Validation d'un workflow

```bash
pnpm val workflow.generated.json
```

Le script `validate` applique le schéma JSON strict (`schemas/n8n-workflow.schema.json`) via [Ajv](https://ajv.js.org/). Toute propriété inconnue provoque un échec.

## Import dans n8n

Le fichier JSON généré peut être importé directement dans l'interface n8n (menu **Workflows > Import from File**).

## Personnalisation

- Modifiez les prompts dans `system/` pour imposer vos conventions internes.
- Ajoutez ou adaptez des exemples dans `src/examples/` pour servir de référence aux tests et à la validation manuelle.
- Enrichissez les few-shots dans `system/FEW_SHOTS.md` pour guider la génération de workflows spécifiques.

## Tests

Des exemples sont fournis dans `src/examples/`. Vous pouvez les valider avec :

```bash
pnpm val src/examples/basic-webhook.json
```

## Structure du projet

```
├─ README.md
├─ system/
│  ├─ SYSTEM_PROMPT.fr.md
│  ├─ DEV_PROMPT.fr.md
│  └─ FEW_SHOTS.md
├─ schemas/
│  └─ n8n-workflow.schema.json
├─ src/
│  ├─ generate.ts
│  ├─ validate.ts
│  ├─ types.ts
│  └─ examples/
│     ├─ basic-webhook.json
│     └─ vision-llm-sheets.json
└─ package.json
```

## Licence

Distribué sous licence MIT.
