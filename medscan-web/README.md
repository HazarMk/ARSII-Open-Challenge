# MedScan Web (React)

## 1) Lancer le backend FastAPI

```powershell
cd "c:\Users\Hazar\Desktop\hackaton ARSii\open chalenge\medscan-ai"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8010 --reload
```

## 2) Lancer le frontend React

```powershell
cd "c:\Users\Hazar\Desktop\hackaton ARSii\open chalenge\medscan-web"
npm install
npm run dev
```

Ouvrir `http://127.0.0.1:5173`.

## Notes

- L'URL backend est definie via `VITE_API_BASE_URL` (sinon fallback `http://127.0.0.1:8010`).
- Exemple PowerShell:
  ```powershell
  $env:VITE_API_BASE_URL="http://127.0.0.1:8010"
  npm run dev
  ```
- L'ecran d'analyse inclut maintenant des champs optionnels: `patient_id`, `exam_type`, `clinical_note`.
- Rapport PDF: bouton `Telecharger rapport PDF` depuis le resultat ou l'historique.
- Triage: `routine`, `high`, `critical` avec filtres sur cas critiques.
- Page `Queue Critique`: liste temps reel des cas urgents avec action `Marquer traite`.
- Auth activee:
  - `medecin / med123`
  - `admin / admin123`
