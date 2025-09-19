Contexte développeur :
- Nœuds disponibles : `n8n-nodes-base.httpRequest`, `n8n-nodes-base.set`, `n8n-nodes-base.if`, `n8n-nodes-base.function`, `n8n-nodes-base.googleSheets`, `n8n-nodes-base.openAi`, `n8n-nodes-base.webhook`.
- Respecte le positionnement : commence à `{ "x": 260, "y": 200 }` pour le premier nœud, puis incrémente `x` de 260 et `y` de 200 pour chaque nœud suivant.
- Toutes les connexions doivent utiliser la sortie `main[0][0]` vers l'entrée principale du nœud suivant.
- Les credentials doivent être référencés avec des placeholders `{{CREDENTIALS.NAME}}` et respecter la regex du schéma.
- Ajoute des notes lorsque la spec impose des détails importants ou des hypothèses.
- Utilise les heuristiques d'atomicité : un nœud par responsabilité métier.
- Prévois un fallback HTTP générique si aucun nœud spécialisé n'existe.
- Normalise les données avec des nœuds Set ou Function avant d'écrire dans des destinations externes.
