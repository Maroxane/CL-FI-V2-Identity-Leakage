# Guide d'utilisation (FR) : extraction, entraînement, audit de fuite et test (RTX 3060)

Code de l'article *Shared-Gate Multi-Branch Fusion of Linguistic, Paralinguistic and Nonverbal Cues for Interpretable Apparent Personality Recognition*.

## Démarrage rapide

1. **Python 3.10, 3.11 ou 3.12** doit être installé (3.11 recommandé, case « Add to PATH » cochée). MediaPipe ne supporte pas Python 3.13.
2. Avoir un **pilote NVIDIA récent** (version 550 ou plus) ; `nvidia-smi` doit fonctionner dans un terminal.
3. Double-cliquer sur **`start.bat`**.
   - Au premier lancement, il installe l'environnement `.venv`, PyTorch CUDA 12.4 et les dépendances (environ 4 Go, 10 à 20 min).
   - Il ouvre ensuite le **suivi de l'entraînement** : http://127.0.0.1:8001
   - L'**interface de test** est sur http://127.0.0.1:8002
4. Dans le tableau de bord, cliquer dans l'ordre sur **1 · Extraire les features**, puis **2 · Entraîner SG-MBF**, puis sur les étapes 3 à 6.
5. **`stop.bat`** arrête les interfaces et la tâche en cours. Les features en cache et les runs terminés sont conservés : chaque étape reprend là où elle s'était arrêtée.

`reinstaller.bat` force la réinstallation des dépendances au prochain `start.bat`.

## Données attendues (dossier `Chalearn`)

Les fichiers sont cherchés **récursivement**, donc l'organisation des sous-dossiers importe peu.

| Fichier | Rôle | Statut actuel |
|---|---|---|
| `*.mp4` (train, validation, test) | vidéos | train partiel (≈ 5 700 clips), validation et test absents |
| `annotation_training.pkl`, `annotation_validation.pkl`, `annotation_test.pkl` | labels Big Five | **absents : indispensables** |
| `transcription_*.pkl` | transcriptions officielles | absentes : Whisper est utilisé à la place (plus lent) |
| `eth_gender_annotations_test.csv` | analyse d'équité | absent : l'étape 6 sera ignorée |

- **Splits officiels** : quand les trois fichiers d'annotations et les vidéos correspondantes sont présents.
- **Split interne 70/15/15** : sinon, sur les clips étiquetés, groupés par vidéo YouTube. Il est signalé en orange dans le tableau de bord, et les résultats ne sont alors **pas comparables** au test officiel.

## Compléter le dataset depuis Hugging Face

Lancer `telecharger_dataset.bat`, ou le bouton **0 · Télécharger les données manquantes** du tableau de bord. Le script :
- télécharge depuis `yeray142/first-impressions-v2` uniquement les clips absents (validation et test, environ 12 Go, plus les clips d'entraînement manquants) dans `Chalearn\hf\`. Le téléchargement reprend là où il s'est arrêté ;
- récupère les métadonnées du dépôt et en tire automatiquement `transcription_<split>.pkl`, `annotation_<split>.pkl` (si les scores Big Five y figurent) et `eth_gender_annotations_test.csv` (si genre et ethnie y figurent). Les fichiers officiels déjà présents ne sont jamais écrasés ;
- indique à la fin s'il manque encore `annotation_validation.pkl` ou `annotation_test.pkl`. Ces labels se trouvent sur la page officielle ChaLearn First Impressions V2 ; il suffit de les copier dans `Chalearn`.

Le dataset est sous licence CC BY-NC 4.0 : usage de recherche non commercial uniquement.

## Données officielles ChaLearn (recommandé pour l'article)

1. Créer un compte gratuit sur https://chalearnlap.cvc.uab.cat/register/ puis se connecter.
2. Télécharger depuis https://chalearnlap.cvc.uab.cat/dataset/24/description/ :
   - **Validation** (`/data/42/files/`) : vidéos, `annotation_validation`, `transcription_validation` ;
   - **Test** (`/data/43/files/`) : `test80_01.zip` à `test80_25.zip`, `annotation_test`, `transcription_test` ;
   - **Training** (`/data/41/files/`) : uniquement si vous voulez remplacer les clips d'entraînement manquants ;
   - **Ethnicity+Gender annotations** (`/data/58/files/`) : `eth_gender_annotations_*.csv`, pour l'étape 6.
3. Laisser les `.zip` dans `SGMBF\Chalearn V2\` (ou `Chalearn\`) puis lancer **`extraire_archives.bat`**. Le contenu est extrait dans `Chalearn\official\` ; les vidéos déjà présentes sont ignorées, donc rien n'est dupliqué. Les archives sont protégées par les mots de passe publiés sur les pages ChaLearn : copiez-les, un par ligne, dans `chalearn_passwords.txt` (fichier exclu du dépôt Git). Les archives imbriquées sont aussi extraites.
4. Relancer **1 · Extraire les features** : seuls les nouveaux clips sont traités. Le projet passe automatiquement aux splits officiels.

## Cross-dataset FI V2 → MuPTA

1. Double-cliquer sur **`extraire_mupta.bat`**. Il extrait `Mupta\audio\speaker_XX.zip` et `Mupta\video\speaker_XX.zip` vers `Mupta\extracted\` (environ 50 Go pour la vidéo). L'espace disque est vérifié avant chaque archive.
   - `Shared.zip` (54 Go) est seulement inspecté : son contenu s'affiche. Ajoutez `--shared` uniquement s'il contient autre chose que les archives par locuteur.
   - Options : `--list` (inventaire seul), `--audio-only`.
2. Bouton **7 · MuPTA : features**. Il traduit les transcriptions russes en anglais avec OPUS-MT, puis calcule les mêmes features que pour FI dans `cache_mupta\`.
3. Bouton **8 · Cross-dataset FI → MuPTA**. Il faut que l'étape 2, et idéalement l'étape 3, soient terminées. Les résultats sont :
   - `results\cross_mupta.csv` : zero-shot de toutes les fusions, sans texte, re-standardisation, recalibration LOSO, fine-tuning de la tête en LOSO ;
   - `results\mupta_gates.csv` : portes moyennes sur MuPTA et sur le test FI ;
   - le Tableau 8 dans `results\tables.md` ;
   - `results\mupta_metadata_test_predictions.csv` : prédictions pour les 6 locuteurs de test, au format `metadata_test.csv`, à envoyer aux auteurs de MuPTA.

Le névrosisme de MuPTA est converti en 1 − N (`mupta.invert_neuroticism`), comme le recommande le readme du corpus. Le signe de la corrélation zero-shot pour ce trait permet de vérifier cette convention.

## Ce que fait chaque étape

| Étape | Commande équivalente | Sorties |
|---|---|---|
| 1 · Extraire | `python -m sgmbf.extract` | `cache/text`, `cache/audio`, `cache/visual` (un `.npz` par clip) |
| 2 · SG-MBF | `python -m sgmbf.experiments --plan main` | `runs/main/…` (5 seeds), `results/seeds.csv`, `gates_test.csv`, `tables.md` |
| 3 · Fusion | `--plan fusion` | 6 baselines sur les mêmes features (Tableau 6), `fusion_seeds.csv` |
| 4 · Ablations | `--plan ablation` | sous-ensembles de modalités et de composantes (Tableaux 4 et 5) |
| 5 · Robustesse | `--plan robustness` | bruit audio 40 → 0 dB, perte d'images, retrait d'une modalité, `robustness.csv` |
| 6 · Équité | `--plan fairness` | `fairness.csv` (Tableau 7) |

Le fichier `results/tables.md` contient les Tableaux 3 à 6 déjà mis en forme (moyenne ± écart-type, IC bootstrap à 95 %, p de Holm). Les figures 4 à 6 de l'article s'obtiennent avec :
`.venv\Scripts\python make_results_figures.py --results results --out figures`

## Optimisations pour la RTX 3060 (6 Go en portable, 12 Go en desktop)

- Une seule famille de modèles en VRAM à la fois : Whisper, puis BERT et S-BERT, puis VGGish, puis Emo-AffectNet. La mémoire est libérée entre chaque.
- Whisper large-v3 tourne via faster-whisper en `int8_float16` (≈ 3 Go). Les DLL CUDA de PyTorch sont exposées à CTranslate2, avec repli automatique sur le CPU si besoin.
- Les encodeurs (BERT, Emo-AffectNet) tournent en fp16. MediaPipe tourne sur `visual_workers` processus CPU en parallèle du GPU, et les images sont réduites à 640 px.
- L'entraînement se fait sur features en cache, entièrement chargées en VRAM, avec AMP fp16 et TF32. Une époque prend moins d'une seconde ; SG-MBF × 5 seeds prend quelques minutes.
- Si la VRAM ou la RAM sont insuffisantes, réduire dans `config.yaml` : `emo_batch` (64 → 16), `text_batch_size` (32 → 8), `visual_workers` (4 → 2).

Durées indicatives pour 10 000 clips sur un portable RTX 3060 (ordre de grandeur, à vérifier sur votre machine) :
- transcription Whisper : 5 à 8 h, ou quelques minutes avec les `transcription_*.pkl` ;
- features visuelles : 2 à 4 h ;
- features audio : environ 1 h ;
- toutes les expériences : 1 à 2 h, plus environ 1 h pour le bruit audio.

## Différences avec le texte de l'article à reporter

- **Empath fournit 194 catégories**, pas 200. La dimension textuelle est donc *D_t* = 768 + 768 + 194 = **1 730** (Tableau 1 et équation 1).
- Le recadrage du visage utilise la boîte englobante des landmarks MediaPipe, et non MTCNN. Remplacez « MTCNN » dans la section 3.4, ou ajoutez MTCNN.
- *D_g* = 33 descripteurs géométriques, *D_b* = 21 descripteurs cinématiques, donc *D_v* = 33 + 1 536 + 21 = **1 590**.
- La baseline « cross-modal transformer » est un encodeur Transformer sur les 3 tokens de modalité : nommez-la ainsi dans les Tableaux 3 et 6.
- Le bruit de robustesse est un bruit blanc. Le bruit « babble » nécessite un fichier wav (`evaluation.babble_wav`).
- Si Emo-AffectNet ne se télécharge pas, un ResNet-50 ImageNet est utilisé en secours. Le tableau de bord l'indique (« Expressions ») : dans ce cas, ne pas publier ces résultats sous le nom Emo-AffectNet.

## Dépannage

- **`CUDA indisponible`** : mettre à jour le pilote NVIDIA, puis lancer `reinstaller.bat` et `start.bat`.
- **Erreur de port** : lancer `stop.bat`, puis `start.bat`.
- **Journaux** : `logs\job.log` (tâches), `logs\job_console.log`, `logs\monitor.log`, `logs\demo.log`.
- **Bouton « Diagnostic »** du tableau de bord : affiche la version de Python et de torch, le GPU et les fichiers du dataset trouvés.
- **Interface de test** : elle charge Whisper, BERT, VGGish et Emo-AffectNet à la première vidéo (20 à 60 s). Le bouton « Libérer la VRAM » les décharge, à utiliser avant de relancer une extraction.
