### Exemple 1 — Webhook → Set → Google Sheets
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Webhook Entrant",
      "type": "n8n-nodes-base.webhook",
      "position": { "x": 260, "y": 200 },
      "parameters": {
        "path": "incoming-lead",
        "httpMethod": "POST",
        "responseMode": "onReceived"
      }
    },
    {
      "id": "2",
      "name": "Nettoyage Données",
      "type": "n8n-nodes-base.set",
      "position": { "x": 520, "y": 400 },
      "parameters": {
        "keepOnlySet": true,
        "values": {
          "string": [
            { "name": "email", "value": "={{$json.body.email}}" },
            { "name": "source", "value": "Webhook" }
          ]
        }
      }
    },
    {
      "id": "3",
      "name": "Ajouter à Sheets",
      "type": "n8n-nodes-base.googleSheets",
      "position": { "x": 780, "y": 600 },
      "parameters": {
        "operation": "append",
        "sheetId": "={{$env.SHEET_ID}}",
        "columns": ["email", "source"]
      },
      "credentials": {
        "googleApi": "{{CREDENTIALS.GOOGLE_API}}"
      }
    }
  ],
  "connections": {
    "Webhook Entrant": { "main": [[{ "node": "Nettoyage Données", "type": "main", "index": 0 }]] },
    "Nettoyage Données": { "main": [[{ "node": "Ajouter à Sheets", "type": "main", "index": 0 }]] }
  },
  "active": false,
  "settings": {},
  "meta": { "template": "lead-intake" },
  "version": 1
}
```

### Exemple 2 — Vision/Transcription → LLM → Sheets
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Webhook Média",
      "type": "n8n-nodes-base.webhook",
      "position": { "x": 260, "y": 200 },
      "parameters": {
        "path": "media-upload",
        "httpMethod": "POST"
      }
    },
    {
      "id": "2",
      "name": "Analyser Média",
      "type": "n8n-nodes-base.openAi",
      "position": { "x": 520, "y": 400 },
      "parameters": {
        "operation": "chat",
        "model": "gpt-4.1-mini",
        "inputType": "json",
        "jsonParameters": true,
        "messages": [
          {
            "role": "system",
            "content": "Tu es un assistant qui décrit précisément le contenu des médias."
          },
          {
            "role": "user",
            "content": "={{$json[\"body\"].file}}"
          }
        ]
      },
      "credentials": {
        "openAiApi": "{{CREDENTIALS.OPENAI_API}}"
      }
    },
    {
      "id": "3",
      "name": "Structurer Résumé",
      "type": "n8n-nodes-base.set",
      "position": { "x": 780, "y": 600 },
      "parameters": {
        "keepOnlySet": true,
        "values": {
          "string": [
            { "name": "titre", "value": "={{$json.summary.title}}" },
            { "name": "description", "value": "={{$json.summary.description}}" }
          ]
        }
      }
    },
    {
      "id": "4",
      "name": "Enregistrer Résumé",
      "type": "n8n-nodes-base.googleSheets",
      "position": { "x": 1040, "y": 800 },
      "parameters": {
        "operation": "append",
        "sheetId": "={{$env.SHEET_ID}}",
        "columns": ["titre", "description"]
      },
      "credentials": {
        "googleApi": "{{CREDENTIALS.GOOGLE_API}}"
      }
    }
  ],
  "connections": {
    "Webhook Média": { "main": [[{ "node": "Analyser Média", "type": "main", "index": 0 }]] },
    "Analyser Média": { "main": [[{ "node": "Structurer Résumé", "type": "main", "index": 0 }]] },
    "Structurer Résumé": { "main": [[{ "node": "Enregistrer Résumé", "type": "main", "index": 0 }]] }
  },
  "active": false,
  "settings": {},
  "meta": { "template": "media-summary" },
  "version": 1
}
```

### Exemple 3 — C4 Agent : Plan → Act → Log (fallback HTTP)
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Déclencheur Plan",
      "type": "n8n-nodes-base.webhook",
      "position": { "x": 260, "y": 200 },
      "parameters": {
        "path": "c4-plan",
        "httpMethod": "POST"
      }
    },
    {
      "id": "2",
      "name": "Plan d'Actions",
      "type": "n8n-nodes-base.function",
      "position": { "x": 520, "y": 400 },
      "parameters": {
        "functionCode": "return [{ actions: $json.body.tasks }];"
      }
    },
    {
      "id": "3",
      "name": "Exécuter Action",
      "type": "n8n-nodes-base.httpRequest",
      "position": { "x": 780, "y": 600 },
      "parameters": {
        "url": "={{$json.actions[0].endpoint}}",
        "method": "POST",
        "jsonParameters": true,
        "sendBody": true,
        "bodyParametersJson": "={{$json.actions[0].payload}}"
      }
    },
    {
      "id": "4",
      "name": "Journaliser Résultat",
      "type": "n8n-nodes-base.googleSheets",
      "position": { "x": 1040, "y": 800 },
      "parameters": {
        "operation": "append",
        "sheetId": "={{$env.SHEET_ID}}",
        "columns": ["endpoint", "status", "payload"]
      },
      "credentials": {
        "googleApi": "{{CREDENTIALS.GOOGLE_API}}"
      },
      "notes": "Fallback HTTP utilisé si aucun connecteur dédié n'est disponible."
    }
  ],
  "connections": {
    "Déclencheur Plan": { "main": [[{ "node": "Plan d'Actions", "type": "main", "index": 0 }]] },
    "Plan d'Actions": { "main": [[{ "node": "Exécuter Action", "type": "main", "index": 0 }]] },
    "Exécuter Action": { "main": [[{ "node": "Journaliser Résultat", "type": "main", "index": 0 }]] }
  },
  "active": false,
  "settings": {},
  "meta": { "template": "c4-agent" },
  "version": 1
}
```
