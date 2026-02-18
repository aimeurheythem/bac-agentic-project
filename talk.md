# 🎤 Bac Agent — Présentation Talk Script
> **Darija Algérienne + Technical English**
> Slide par slide — parle naturellement, khelli rahat wach

---

## 🟦 Slide 1 — Cover

*[oqef shwiya, khelli les gens yshofou l'écran]*

---

**"Bonjour tout le monde —**
lyom ghadi nchoufou plan Ta3 **Bac Agent** —
AI platform mbniya khassatan l**students ta3 baccalauréat** fi l'Algérie.

L'idea simple — kol student fi l'Algérie ykhdem fel bac, w maandouch ai tool yfahmou w y3awnouh haq
machi tool 3adi, hada tool **yfahem filière diyalk, mwad diyalk, w coefficient diyak.**

Hadchi li ghadi nchoufou lyoum : kifech rah nebno l platform, l architecture, l AI modes,
w kifach l student ighadi yt3amal m3aha."

---

## 🟦 Slide 2 — What is Bac Agent?

---

**"Khassni n3tikom picture wach kayen w wach mkayen.**

**Wach kayen darwek ?**
Students bezzaf yekhdmou via Google, vai ChatGPT — w ChatGPT ma3rafch wach hiya **filière Sciences Expérimentales**,
ma3rafch **coefficient dyal matière**, w ma3rafch kif yet3amal **barème** fi bac ta3 Algérie.

**7 filières officielles** — mathématiques, sciences, technique, gestion...
**5 AI modes** — kol mode 3andou but mkhtalef.
**10 ans d'examens** — 2015 l 2024 — **real bac exams** — kolhom mprocessés w fi system.

**Bac Agent** machi juste chatbot — howa tutor y3ref dossier dyal l'étudiant."

---

## 🟦 Slide 3 — 7 Official Streams

---

**"Hna l challenge l awal —**
fi l'Algérie makaynach filière wahda — kayen **7 filières**, w kollwahed 3andou **mwad** w **coefficients** mkhtalfin.

Mathématiques — coefficient dyal math **7** — hia la matière l kbira.
Sciences Expérimentales — sciences natur **6** — physics **5**.
Technique Mathématique — **la plus complexe** — 4 options : Civil, Mécanique, Électrique, Procédés —
kol option 3andha **curriculum** khass biha w **coefficient table** mkhtalfa.

**L challenge** ki dorna : l AI lazem y3ref wach hiya filière dyal l'étudiant, w specialty ta3o,
w ybeddel **system prompt** lel automatique — machi manuellement."

---

## 🟦 Slide 4 — System Architecture

---

**"darwek nchoufou kif mbniya l system.**

*[chir l'écran]*

Fel foq — **Student** — ykhdem m3a **React frontend**.
Yeb3ath request l **FastAPI backend** — Python 3.11.
Mel backend, kayen 3 services li khdmou m3a ba3dhom :

**Premier** — **Tutor Agent** — hada howa l cerveau dyal system — howa li ychawer m3a GPT-4o.
**Deuxième** — **RAG Pipeline** — hada li yjib l context mel examens réels dyal bac.
**Troisième** — **OCR Engine** — hada li convertit les PDFs dyal examens l texte w LaTeX.

Fel base — **SQLite database** — fiha les streams, les matières, w les coefficients.
**Vector Store** — fiha les chunks dyal examens — mprocessés w mconvertis l embeddings.
**PDF Exams** — 2015 l 2024 — kolhom mstockés w mprocessés.

**Kollchi yet3amel me3a** — l student yeb3ath question, l system yjib context mel examens,
l tutor agent y3awed yjeml w yji b jawab précis."

---

## 🟦 Slide 5 — Technology Stack

---

**"w drwek les outils li rah nakhdmna bihom :**

**Frontend** — React 18 + TypeScript — build bih Vite — **fast, responsive, RTL** pour l'arabe.
**Backend** — FastAPI — Python — **async** — kol endpoint ykhdem b les requêtes en parallèle.
**AI / LLM** — OpenAI GPT-4o — hada l model l akhir.
**Embeddings** — OpenAI text-embedding-3-small — pour l **vector search** dyal RAG.
**Database** — SQLite local — w fel production ghadi nstorni l PostgreSQL.
**OCR** — 3 providers : Mathpix pour les formules, Google Vision, w Tesseract fallback.

**Stack mkhtar b 3aqel** — kol piece fiha raison."

---

## 🟦 Slide 6 — The Tutor Agent & 5 Chat Modes

---

**"Hna l cœur dyal system — l Tutor Agent.**

kamel les requests ta3 student ydirou mel **TutorAgent class** —
w hada l agent y3awed yjma3 **system prompt dynamique** men 4 parties :
context dyal filière, behavior dyal mode, w RAG context mel examens réels.

**Les 5 modes :**

**Orientation** — *general mode* —
machi l agent ychrah concepts w ychall exercises —
hado li ygdar y3awnek tkhetet **jadwal drassi**, twajahak b les matières prioritaires,
w tfekker fel bac men zaqat view strategique.

**Exercises** — *exercise_help* —
machi agent yjeblak l jawabDirectement —
ychawer m3ak comme un vrai prof — yas2alk awel, ybeyen l erreur, w yjib l réponse bass b3d ma t7awel.

**Concepts** — *concept_explanation* —
structure fixe : Définition, Théorème, Intuition, Formule, Exemples, Misconceptions —
comme un cours complet fel response wahda.

**Exam Prep** — *exam_prep* —
yrekzou 3la les examens réels dyal 2015 l 2024 —
y3erfak les **patterns** li tet3awd, kifach tdber waqtek, w chwiya tactic.

**Review** — *solution_review* —
nta kteb l solution diyalk — l agent ychek kol khTwa khTwa
w y3tik score selon le **barème officiel**."

---

## 🟦 Slide 7 — RAG Pipeline

---

**"Wach howa RAG w 3lach important ?**

**RAG = Retrieval-Augmented Generation** —
behal ma t3tih l GPT context réel men les examens bac — zdad mel knowledge générale diyah.

**Kif khedmna bih :**

**Awel** — n7ottou kolhom lessexamens PDFs — 2015 l 2024 — fel **OCR engine**.
**Tani** — n3emelou **chunking** — nqes3ou les textes l parties sahghira —
kayen 4 strategies : lessons, exercises, solutions, w general.
**Talt** — kol chunk n7awlouhou l **embedding** — vector — khdem bih OpenAI.
**Rab3** — nstockiwou kolhom fel **vector store** — chunks.json + embeddings.npy.
**Khames** — ki l student y9essek — n3emlu **cosine similarity search** —
njibu top les chunks les plus proches.
**Sades** — n7ottou l context hadak fel **system prompt** — w l GPT yjaweb b des références réelles.

**Exemple pratique** : student y9essek 3la l limite dyal une fonction —
l system yjib chunks mel bac 2019 dyal math — w l agent yjeml w yjiblak jawab m3a référence réelle."

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

**Coefficient engine** — feature spéciale —
l étudiant y9der ydakhel les notes diyah —
l system y7eseb **moyenne pondérée** b les coefficients officiels —
w y3tih la mention : Passable, Assez Bien, Bien, Très Bien."

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
Nbniw l OCR engine — convertit les PDFs d'examens l texte w LaTeX.
Nprocessiw les exam files 2015 l 2023.

**Phase 2 — MVP AI + Frontend :**
Nbniw le RAG pipeline complet — embeddings w retrieval.
Nbniw le Tutor Agent m3a les 5 chat modes.
Nbniw le React SPA — onboarding + chat flows.
N9adiw inline chat w history sidebar.

**Phase 3 — Advanced Features :**
N9adiw image input — l'étudiant y9der ysowwer exercise w yb3ath la photo.
Mode timed Mock Exam m3a auto-scoring.
Voice support pour les matières شفوية.
Migration l Supabase pgvector pour la production.

**Phase 1 w 2 — hadou mbenyin** — ghadi nchoufouhoum live.
**Phase 3** — hado les fonctionnalités li ghadi n9adiwhoum."

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
