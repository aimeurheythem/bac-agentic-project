# 🎤 Bac Agent — Présentation Talk Script
> **Darija Algérienne + Technical English**
> Slide par slide — parle naturellement, khelli rahat wach

---

## 🟦 Slide 1 — Cover,

---

**"salam 3likom tout le monde —**
lyom rah nchoufou plan Ta3 **Bac Agent** —
AI platform mbniya l**students ta3 baccalauréat** fi l'Algérie.

L'idea simple — kol student fi l'Algérie ykhdem fel bac, w maandouch ai tool yfahmou w y3awnou haq
machi tool normal, hada tool **yfahem filière diyalk, mawad diyalk, w coefficient dyalek.**

wesh rah nchoufou lyoum : kifech rah nebno l platform, l architecture ta3ha,wel AI modes li rah nbnohom,
w kifach l student rah yt3amal m3aha."

---
1
## 🟦 Slide 2 — What is Bac Agent?

---

**"Khassni n3tikom picture wach kayen w wach mkayen.**

**Wach kayen darwek ?**
Students bezzaf yekhdmou via Google, vai ChatGPT — w ChatGPT ma3rafch wach hiya **filière Sciences Expérimentales**,
ma3rafch **coefficient dyal matière**, w ma3rafch kif yet3amal **barème** fi bac ta3 Algérie.

**7 filières officielles** — mathématiques, sciences, technique, gestion...
**5 AI modes** — kol mode 3andou but mkhtalef.
**10 ans ou plus d'examens** — 2012 l 2025 — **real bac exams** — kolhom mprocessés w fi system.

**Bac Agent** machi juste chatbot — howa tutor (ostad) y3ref dossier dyal l'étudiant."

---

## 🟦 Slide 3 — 7 Official Streams

---

**"Hna l challenge l awal —**
fi l'Algérie makaynach filière wahda — kayen **7 filières**, w kollwahed 3andou **mwad** w **coefficients** mkhtalfin.

par exemple filière Mathématiques — coefficient ta3 math **7** — hia la matière l kbira.

Sciences Expérimentales — sciences natur **6** — physics **5**.
Technique Mathématique — **la plus complexe** — 4 options : Civil, Mécanique, Électrique, Procédés —

kol option 3andha **minhaj** khass biha w **coefficient table** mkhtalfa.

**L challenge** ki dorna : l AI lazem y3ref wach hiya filière dyal l'étudiant, w specialty ta3o,
w ybeddel **system prompt** lel automatique — machi manuellement."

---

## 🟦 Slide 4 — System Architecture

---

**"darwek nchoufou kifh nbno l system.**


Fel foq — **Student** — yinteragier m3a L'UI ta3na.
Yeb3ath request l **backend** ta3na li mebni bel FastAPI.
Mel backend, kayen 3 services li khdmou m3a ba3dhom :

**Premier** — **Tutor Agent** — hada howa l cerveau ta3 system — howa li yahder m3a LLM ta3na.

**Deuxième** — **RAG Pipeline** — hada li yjib l context mel examens ta3 bac.

**Troisième** — **OCR Engine** — hada li convertit les PDFs ta3 l'examens l texte.

Fel base — **SQLite database** — fiha les streams, les matières, w les coefficients.
**Vector Store** — fiha les chunks ta3 les examens — mprocessés w mconvertis l embeddings.
**PDF Exams** — 2012 l 2024 — kolhom mstockés w mprocessés.

**Donc** — student yeb3ath question, system yjib context mel examens, tutor agent yjemel kolchi w yjib response précis."

---

## 🟦 Slide 5 — Technology Stack

---

**"les outils li rah nkhdmo bihom :**

**Frontend** — React 18 + TypeScript — build bih Vite — **fast, responsive, RTL** pour l'arabe.
**Backend** — FastAPI — Python — **async** — kol endpoint ykhdem b les requêtes en parallèle.
**AI / LLM** — OpenAI GPT-4o — hada l model l akhir.
**Embeddings** — OpenAI text-embedding-3-small — pour l **vector search** dyal RAG.
**Database** — SQLite local — w fel production ghadi nstorni l PostgreSQL.
**OCR** — 3 providers : Mathpix pour les formules, Google Vision, w Tesseract.

---

## 🟦 Slide 6 — The Tutor Agent & 5 Chat Modes

---

**"Hna l cœur dyal system — l Tutor Agent.**

**Les 5 modes :**

**Orientation** — *general mode* —
hada li ygdar y3awnek tkhetet **jadwal drassi**, ywajhak b les matières prioritaires, w y3tik des conseils.

**Exercises** — *exercise_help* —
ychawer m3ak comme un vrai prof — yas2alk awel, ybeyen l erreur, w yjib l réponse b3d ma t7awel.

**Concepts** — *concept_explanation* —
Définition, Théorème, Intuition, Formule, Exemples, Misconceptions — comme un cours complet fel response wahda.

**Exam Prep** — *exam_prep* —
hna nrekzou 3la les examens dyal bac ywerilk par exemple les **patterns** li tet3awd, kifach tdber waqtek, w chwiya tactic.

**Review** — *solution_review* —
nta kteb l solution diyalk — l agent ydir chek l solution ta3ek khTwa khTwa w y3tik score selon le **barème officiel**."

---

## 🟦 Slide 7 — RAG Pipeline

---

**"Wach howa RAG w 3lach important ?**

**RAG = Retrieval-Augmented Generation** —
Ya3ni nmeddou l'GPT context réel men les examens ta3 l'bac — zyada 3la l'knowledge générale li 3andou déjà.

**Kif khedmna bih :**

**Awel** — n7ottou kamel les examens PDFs — 2015 l 2024 — fel **OCR engine**.
**Tani** — ndiro **chunking** — nqasmo les textes l des parties sghar —
kayen 4 strategies : lessons, exercises, solutions, w general.
**Talt** — kol chunk n7awlouhou l **embedding** — vector — ykhdem bih OpenAI.
**Rab3** — nstockiwhoum kamel fel **vector store** — chunks.json + embeddings.npy.
**Khames** — ki l student ysa9si — ndiro **cosine similarity search** —
njibou top les chunks les plus proches.
**Sades** — n7ottou l context hadak fel **system prompt** — w l GPT yjaweb b des références réelles.

**Exemple pratique** : student ysa9si 3la l limite ta3 une fonction —
l system yjib chunks mel bac 2019 ta3 math — w l agent yejma3 kolchi w yjiblak jawab m3a référence réelle."

---

## 🟦 Slide 8 — Student Journey

---

**"Ola nchoufou kif l student yt3amal m3a l platform.**

*[chir l'écran — l'UX flow diagram]*

**1 — Onboarding** —
l awel chy l student ydkhol — ykhter **filière** diyah —
w kun fi **technique math**, ykhter specialty : Civil, Mécanique, etc.
Hadchi ytsave — w l system yb9a y3ref 3lih tout le long.

**2 — Dashboard** —
ydkhol l dashboard — yshof **5 mode chips** fel foq dyal l input box —
ykhter l mode, ykteb l question, w y9essek.

**3 — AI Reply** —
l response tji **markdown + LaTeX** rendered —
les formules maths tji mformatées nickel — machi tekste 3adi.

**4 — History sidebar** —
kayen sidebar 3la l ymin —
fiha kol les conversations dyal l session — yqder yrja3 li ay conversation ma bghach.

**5 — New Chat** —
y9der ybda chat jdid b click wahda — state tresat w table rase."

---

## 🟦 Slide 9 — Data Models

---

**"Deba nchoufou kif morganisés les données.**

**4 tables l assasyin :**

**streams** — les 7 filières — code, nom, nom_ar, w has_options pour technique math.
**subjects** — les matières — code, catégorie, nom.
**coefficients** — hada l plus important — yrabt bin filière w matière —
w kayen specialty_option pour tech math —
kol combination 3andha coefficient specific.
**users** — l'étudiant — email, filière, specialty.


---

## 🟦 Slide 10 — Key API Endpoints

---

**"Voilà les principales APIs :**

**GET /streams** — yjib les 7 filières avec les noms en arabe.
**GET /streams/{id}/specialties** — yjib les options dyal technique math.
**POST /calculate-average** — l coefficient engine — ydakhel notes, ytla3 la moyenne.
**POST /chat** — hada l endpoint l principal — ybeth message l tutor agent — m3a mode w filière.
**POST /search-context** — RAG search — ydakhel query, ytla3 top-k chunks mel examens.
**GET /subjects** — liste des matières avec filtre.

**Kollha async** — FastAPI — **documented automatiquement** mel Swagger UI."

---

## 🟦 Slide 11 — Implementation Phases

---

**"Hna l roadmap dyal mashrou3 — 3 phases :**

**Phase 1 — Data & Foundation :**
Ghadi nbniw l coefficient engine pour les 7 filières.
Nseediw la base de données avec les streams, matières, w coefficients.
Nbniw l OCR engine — convertit les PDFs d'examens l texte.
Nprocessiw les exam.

**Phase 2 — MVP AI + Frontend :**
Nbniw le RAG pipeline complet — embeddings w retrieval.
Nbniw le Tutor Agent m3a les 5 chat modes.
Nbniw le React SPA (single-page application) — onboarding + chat flows.
najoutiw inline chat w history sidebar.

**Phase 3 — Advanced Features :**
najoutiw image input — l'étudiant y9der ysowwer exercise w yb3ath la photo.
Mode timed Mock Exam m3a auto-scoring.
Voice support.
Migration l Supabase pgvector pour la production.

---

## 🟦 Slide 12 — Closing

---

**"W fin tla3na lyoum :**

*[chir l'écran — closing slide]*

Hada howa Bac Agent —
7 filières, 5 AI modes, RAG 3la 10 ans d'examens réels, OCR engine, coefficient calculator,
w UX mbeniya m3a l étudiant Algérien fi balich.

**Build everything that's shown — w l application tkoun ready to ship.**

Shokran 3la waqtkom — w ana dima disponible l ay question."

---

## 🟦 Q&A Tips
> *Si kayen chy sual li ma3raftouch — 9ol haka :*

- *"Bonne question — hada shay kayen fi roadmap dyal phase 3"*
- *"Deba architecture moptimisée pour MVP — production scale ghadi ythem m3a pgvector"*
- *"L model ghadi ybeddel — system mbniya modular — GPT-4o yqder ytbaddal b Claude aw Mistral"*

---

> 🗒️ **Timing suggéré :** ~15–20 minutes talk + 5–10 minutes Q&A
> ⚡ **Conseil :** chir live demo beynhom slides 8 w 9 — ouvre l app, khter filière, eb3ath question.
