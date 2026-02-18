# Bac-Tutor Frontend

Interface React pour l'application Bac-Tutor - Assistant IA pour le Baccalauréat Algérien.

## 🚀 Démarrage Rapide

### Prérequis
- Node.js 18+
- npm ou yarn

### Installation

```bash
cd admin-panel/client
npm install
```

### Configuration

Créez un fichier `.env` à la racine du dossier client:

```bash
cp .env.example .env
```

Modifiez les variables selon votre environnement:
- `VITE_API_URL`: URL de l'API backend (par défaut: http://localhost:8000)

### Lancer l'application

```bash
# Mode développement
npm run dev

# Build pour production
npm run build

# Preview du build
npm run preview
```

## 📁 Structure du Projet

```
src/
├── components/           # Composants réutilisables
│   ├── onboarding/      # Composants de l'onboarding
│   ├── ErrorBoundary.tsx
│   └── LoadingSpinner.tsx
├── pages/               # Pages principales
│   ├── Onboarding.tsx   # Sélection de filière
│   ├── Dashboard.tsx    # Tableau de bord
│   ├── BacCalculator.tsx # Simulateur de moyenne
│   └── ChatInterface.tsx # Interface de chat IA
├── store/               # Gestion d'état (Zustand)
│   └── appStore.ts
├── lib/                 # Utilitaires et API
│   └── api.ts
├── types/               # Types TypeScript
│   └── index.ts
├── App.tsx              # Composant principal
└── main.tsx             # Point d'entrée
```

## 🛣️ Routes

- `/onboarding` - Sélection de la filière et spécialité
- `/dashboard` - Tableau de bord principal
- `/calculator` - Simulateur de moyenne Bac
- `/chat` - Discussion avec l'IA

## 🎨 Fonctionnalités

### 1. Onboarding
- Sélection parmi 7 filières du Bac
- Spécialités pour Technique Math
- Persistance dans localStorage

### 2. Simulateur de Moyenne
- Formulaire dynamique basé sur la filière
- Calcul en temps réel
- Affichage de la mention (Très Bien, Bien, etc.)
- Détail par matière

### 3. Chat IA
- Support Markdown et LaTeX
- Historique des conversations
- Indicateur de contexte RAG
- Interface responsive

## 🔧 Technologies

- **React 18** - Framework UI
- **TypeScript** - Typage statique
- **Vite** - Build tool rapide
- **React Router** - Navigation
- **Zustand** - Gestion d'état
- **Axios** - Requêtes HTTP
- **React Markdown** + **KaTeX** - Rendu Markdown et mathématiques
- **Lucide React** - Icônes

## 🧪 Tests

```bash
# Lancer les tests
npm run test

# Tests en mode watch
npm run test:watch
```

## 📦 Déploiement

### Build Production

```bash
npm run build
```

Le build sera généré dans le dossier `dist/`.

### Déploiement Vercel

```bash
# Installer Vercel CLI
npm i -g vercel

# Déployer
vercel --prod
```

### Variables d'Environnement Production

Assurez-vous de configurer:
- `VITE_API_URL`: URL de votre API backend en production

## 🤝 Contribution

1. Fork le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📝 License

Ce projet est sous licence MIT.

## 🐛 Débogage

### Problèmes courants

**L'API ne répond pas:**
- Vérifiez que le backend est démarré sur le port 8000
- Vérifiez la variable `VITE_API_URL` dans `.env`

**Erreurs CORS:**
- Le backend doit autoriser les requêtes depuis `http://localhost:5173`
- Vérifiez la configuration CORS dans `backend/main.py`

**Build échoue:**
- Assurez-vous d'avoir Node.js 18+
- Supprimez `node_modules` et réinstallez: `rm -rf node_modules && npm install`

## 📞 Support

Pour toute question ou problème, veuillez ouvrir une issue sur GitHub.
