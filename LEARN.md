# 🎓 MIDI AI Factory - Guide d'Apprentissage

*Un guide complet pour comprendre, maîtriser et étendre ce projet d'IA musicale*

---

## Table des Matières

1. [Présentation du Projet](#1-présentation-du-projet)
2. [Les Données MIDI : De la Musique aux Nombres](#2-les-données-midi--de-la-musique-aux-nombres)
3. [Le Modèle IA : Un Cerveau Musical](#3-le-modèle-ia--un-cerveau-musical)
4. [L'Entraînement sur Apple Silicon](#4-lentraînement-sur-apple-silicon)
5. [La Génération : De l'IA à la Musique](#5-la-génération--de-lia-à-la-musique)
6. [Décortiquons le Code](#6-décortiquons-le-code)
7. [Extensions et Améliorations](#7-extensions-et-améliorations)
8. [Ressources pour Aller Plus Loin](#8-ressources-pour-aller-plus-loin)

---

## 1. Présentation du Projet

### 🎯 L'Objectif

Ce projet crée un système d'IA capable de :
1. **Apprendre** des patterns musicaux à partir de fichiers MIDI
2. **Comprendre** les structures musicales (mélodies, accords, rythmes)
3. **Générer** de nouvelles compositions originales

### 🏗️ L'Architecture Globale

```
Fichiers MIDI → Tokenisation → Modèle Transformer → Génération → Nouveaux MIDI
```

C'est similaire à GPT pour le texte, mais adapté à la musique !

### 💡 Pourquoi ce Projet est Parfait pour Apprendre l'IA

- **Données concrètes** : Les MIDI sont faciles à visualiser et comprendre
- **Résultats tangibles** : Tu peux écouter ce que ton IA génère
- **Architecture moderne** : Tu apprends les Transformers (base de GPT, BERT, etc.)
- **Optimisation hardware** : Tu exploites ton Mac au maximum

---

## 2. Les Données MIDI : De la Musique aux Nombres

### 📊 Qu'est-ce qu'un Fichier MIDI ?

Un fichier MIDI n'est pas de l'audio, c'est une **partition numérique** qui contient :
- **Notes** : quelle touche de piano (0-127)
- **Vélocité** : force de frappe (0-127)
- **Timing** : quand jouer la note
- **Durée** : combien de temps la tenir

```python
# Exemple de données MIDI brutes
Note(pitch=60, velocity=80, start=0.0, end=0.5)  # Do (C4) pendant 0.5s
Note(pitch=64, velocity=75, start=0.5, end=1.0)  # Mi (E4) ensuite
```

### 🔄 Le Pipeline de Traitement

#### 1. **Analyse** (`midi_analyzer.py`)

```python
def analyze_file(self, filepath):
    midi = pretty_midi.PrettyMIDI(filepath)
    
    # Extraire les métadonnées
    tempo = midi.estimate_tempo()
    duration = midi.get_end_time()
    
    # Analyser les notes
    for instrument in midi.instruments:
        for note in instrument.notes:
            # Collecter pitch, velocity, timing...
```

**Pourquoi ?** Pour comprendre ton dataset :
- Tempos moyens (pour générer dans des BPM cohérents)
- Durées typiques (pour créer des clips de bonne longueur)
- Distribution des notes (pour éviter de générer hors tessiture)

#### 2. **Tokenisation** (`midi_tokenizer.py`)

C'est l'étape cruciale : transformer la musique en "mots" que l'IA comprend.

**J'ai choisi REMI+ (REpresentation of Music Improved)** car :
- Il capture le **tempo** (important pour le groove)
- Il encode les **accords** (pas juste des notes isolées)
- Il gère les **silences** (la musique respire !)
- Il préserve la **structure temporelle**

```python
# Avant tokenisation (données MIDI)
Note(pitch=60, velocity=80, start=0.0, end=0.5)

# Après tokenisation (tokens REMI)
["Tempo_120", "Position_0", "Pitch_60", "Velocity_80", "Duration_48"]
```

**Analogie Node.js** : C'est comme parser du JSON en objets JavaScript, mais pour la musique !

### 🎯 Pourquoi Segmenter en 16 Secondes ?

```python
def _segment_tokens(self, tokens, bars_per_segment=64):
    # 64 bars ≈ 16 secondes à 120 BPM
```

- **Mémoire GPU** : Des séquences trop longues = out of memory
- **Apprentissage** : Plus facile d'apprendre des patterns courts
- **Génération** : 16s est une bonne longueur pour un loop musical

---

## 3. Le Modèle IA : Un Cerveau Musical

### 🧠 L'Architecture Transformer

J'ai choisi un **Music Transformer** car c'est l'état de l'art pour les séquences :

```python
class MusicTransformer(nn.Module):
    def __init__(self, 
                 vocab_size=512,      # Taille du vocabulaire de tokens
                 d_model=512,         # Dimension des embeddings
                 n_heads=8,           # Têtes d'attention
                 n_layers=6,          # Couches Transformer
                 d_ff=2048,           # Dimension feed-forward
                 max_seq_len=2048):   # Longueur max des séquences
```

### 🔍 Comment ça Marche ?

#### 1. **Embedding Layer**
```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```
Transforme chaque token (ex: "Pitch_60") en vecteur de 512 dimensions.

**Analogie** : C'est comme Word2Vec mais pour les notes de musique !

#### 2. **Positional Encoding**
```python
class RelativePositionalEncoding(nn.Module):
    # Ajoute l'information de position temporelle
```

**Pourquoi ?** Le Transformer ne sait pas naturellement que "Position_96" vient après "Position_0". On doit lui dire !

#### 3. **Multi-Head Attention**
```python
self.attention = nn.MultiheadAttention(d_model, n_heads)
```

C'est le cœur du modèle. Chaque "tête" regarde différents aspects :
- Tête 1 : "Quelle note suit généralement un Do ?"
- Tête 2 : "Quel accord s'enchaîne bien après Am ?"
- Tête 3 : "Quel pattern rythmique est cohérent ?"
- etc.

#### 4. **Feed-Forward Network**
```python
self.ff = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.GELU(),              # Activation smooth
    nn.Dropout(dropout),     # Régularisation
    nn.Linear(d_ff, d_model)
)
```

Apprend des transformations complexes des patterns musicaux.

### 🎯 Le Masque Causal

```python
def generate_square_subsequent_mask(self, size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
```

**Crucial** : Empêche le modèle de "tricher" en regardant le futur pendant l'entraînement.

```
Position:  0  1  2  3  4
Token 0:   ✓  ✗  ✗  ✗  ✗   (ne voit que lui-même)
Token 1:   ✓  ✓  ✗  ✗  ✗   (voit 0 et 1)
Token 2:   ✓  ✓  ✓  ✗  ✗   (voit 0, 1 et 2)
```

---

## 4. L'Entraînement sur Apple Silicon

### 🚀 Metal Performance Shaders (MPS)

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
```

**Pourquoi c'est génial ?**
- Utilise le GPU intégré de ton Mac
- 10x plus rapide que le CPU
- Pas besoin de NVIDIA !

### 📈 La Boucle d'Entraînement

```python
def train(self, train_loader, val_loader=None, epochs=50):
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            # 1. Forward pass
            logits = self.model(x)
            
            # 2. Calcul de la loss
            loss = self.criterion(logits, y)
            
            # 3. Backward pass
            loss.backward()
            
            # 4. Mise à jour des poids
            self.optimizer.step()
```

### 🎯 Les Hyperparamètres Clés

```python
# Learning rate avec scheduler
self.optimizer = torch.optim.AdamW(lr=1e-4)
scheduler = CosineAnnealingLR(...)  # Diminue progressivement

# Gradient clipping
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
```

**Pourquoi ?**
- **AdamW** : Meilleur que SGD pour les Transformers
- **Cosine Annealing** : Évite de rester bloqué dans des minima locaux
- **Gradient Clipping** : Empêche l'explosion des gradients

### 💾 Checkpointing Intelligent

```python
if val_loss < best_loss:
    self.save_checkpoint(epoch, val_loss, is_best=True)
```

Sauvegarde automatique du meilleur modèle !

---

## 5. La Génération : De l'IA à la Musique

### 🎲 Stratégies de Sampling

#### 1. **Temperature**
```python
logits = logits / temperature
```

- `temperature = 0.5` : Conservateur, répétitif
- `temperature = 1.0` : Équilibré
- `temperature = 1.5` : Créatif, parfois chaotique

**Analogie** : C'est comme le niveau de "folie créative" de ton IA !

#### 2. **Top-K Sampling**
```python
if top_k > 0:
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
```

Ne garde que les K tokens les plus probables. Évite les choix absurdes.

#### 3. **Top-P (Nucleus) Sampling**
```python
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > top_p
```

Plus dynamique que Top-K : s'adapte à la certitude du modèle.

### 🎼 Styles de Génération

```python
def _get_start_tokens(self, style, tempo):
    if style == "chord":
        base_tokens.extend([
            "Position_0",
            "Chord_C:maj",
            "Position_96",
            "Chord_Am:min"
        ])
```

**L'idée** : On "amorce" la génération avec des tokens typiques du style voulu.

---

## 6. Décortiquons le Code

### 📄 `main.py` - Le Chef d'Orchestre

```python
def main():
    # Parse les arguments
    parser = argparse.ArgumentParser()
    
    # Vérifie l'environnement
    if not check_environment():
        sys.exit(1)
    
    # Lance les étapes selon la commande
    if args.command == 'all':
        analyzer.analyze_all()      # 1. Analyse
        tokenizer.process_dataset() # 2. Tokenise
        train.main()               # 3. Entraîne
        generator.generate()       # 4. Génère
```

**Design Pattern** : Command Pattern - chaque action est indépendante.

### 🧮 `model.py` - Le Cerveau

Points clés du code :

1. **Initialisation des poids**
```python
def _init_weights(self):
    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
```
Xavier init = meilleure convergence pour les réseaux profonds.

2. **Génération autoregressive**
```python
for _ in range(max_length - tokens.shape[1]):
    logits = self(tokens)
    next_token = sample(logits[:, -1, :])
    tokens = torch.cat([tokens, next_token], dim=-1)
```
Génère token par token, comme GPT !

### 🎯 `train.py` - L'Entraîneur

Structure modulaire :
```python
class Trainer:
    def __init__(self, model_config, training_config):
        self.device = self._setup_device()
        self.model = MusicTransformer(**model_config)
        self.optimizer = self._setup_optimizer()
    
    def train(self, train_loader, val_loader):
        # Boucle d'entraînement
    
    def validate(self, val_loader):
        # Évaluation
    
    def save_checkpoint(self, epoch, loss):
        # Sauvegarde
```

**Pattern** : Separation of Concerns - chaque méthode a une responsabilité.

### 🎵 `generate.py` - Le Compositeur

La conversion tokens → MIDI :
```python
def _tokens_to_midi(self, tokens, tempo):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    for token in tokens:
        if token_name.startswith("Pitch_"):
            pitch = int(token_name.split("_")[1])
            # Créer la note MIDI
        elif token_name.startswith("Position_"):
            current_position = int(token_name.split("_")[1])
            # Mettre à jour la position temporelle
```

**Astuce** : On parse les noms de tokens pour reconstruire la musique !

---

## 7. Extensions et Améliorations

### 🚀 Idées pour Faire Évoluer le Projet

#### 1. **Multi-Instruments**
```python
# Actuel : un seul instrument
instrument = pretty_midi.Instrument(program=0)

# Amélioration : multi-pistes
instruments = {
    'piano': pretty_midi.Instrument(program=0),
    'bass': pretty_midi.Instrument(program=33),
    'drums': pretty_midi.Instrument(program=128, is_drum=True)
}
```

#### 2. **Conditional Generation**
```python
class ConditionalMusicTransformer(MusicTransformer):
    def __init__(self, ..., n_conditions=8):
        super().__init__(...)
        self.condition_embedding = nn.Embedding(n_conditions, d_model)
    
    def forward(self, x, condition=None):
        if condition is not None:
            cond_emb = self.condition_embedding(condition)
            x = x + cond_emb
```

Permet de générer dans un style spécifique !

#### 3. **Attention Visualization**
```python
def visualize_attention(model, tokens):
    _, attention_weights = model.attention(tokens, return_weights=True)
    
    # Heatmap des poids d'attention
    plt.imshow(attention_weights[0].cpu())
    plt.colorbar()
```

Comprendre ce que "regarde" ton modèle !

#### 4. **MIDI to Audio**
```python
# Utiliser FluidSynth
from midi2audio import FluidSynth

fs = FluidSynth()
fs.midi_to_audio('generated.mid', 'output.wav')
```

#### 5. **Real-time Generation**
```python
class RealtimeGenerator:
    def __init__(self, model, buffer_size=128):
        self.model = model
        self.buffer = []
    
    def generate_next(self, context):
        # Génère le prochain token en temps réel
        with torch.no_grad():
            next_token = self.model.generate(context, max_length=1)
        return next_token
```

### 🎨 Améliorations Architecture

#### 1. **Relative Attention**
```python
class RelativeMultiheadAttention(nn.Module):
    # Meilleure capture des patterns répétitifs en musique
```

#### 2. **Mixture of Experts**
```python
class MoELayer(nn.Module):
    def __init__(self, n_experts=4):
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_experts)
        ])
        self.router = nn.Linear(d_model, n_experts)
```

Différents "experts" pour différents styles musicaux !

#### 3. **VAE pour Plus de Créativité**
```python
class MusicVAE(nn.Module):
    def encode(self, x):
        # Encode en espace latent
        return mu, log_var
    
    def decode(self, z):
        # Décode depuis l'espace latent
        return reconstruction
```

### 📊 Métriques d'Évaluation

```python
def evaluate_generation(generated_midi, reference_midis):
    metrics = {
        'pitch_histogram_distance': compare_pitch_distributions(),
        'rhythm_consistency': analyze_rhythm_patterns(),
        'harmonic_coherence': check_chord_progressions(),
        'originality_score': measure_uniqueness()
    }
    return metrics
```

---

## 8. Ressources pour Aller Plus Loin

### 📚 Livres Essentiels

1. **"Deep Learning" - Ian Goodfellow**
   - LA bible du deep learning
   - Commence par les chapitres sur les RNN/LSTM avant les Transformers

2. **"The Deep Learning Book" - François Chollet**
   - Plus accessible, avec des exemples Keras/TensorFlow

### 🎓 Cours en Ligne

1. **fast.ai - Practical Deep Learning**
   - Approche top-down parfaite pour les développeurs
   - Gratuit et très pratique

2. **CS231n Stanford - Computer Vision**
   - Pour comprendre les CNN (utiles pour les spectrogrammes audio)

3. **Hugging Face Course**
   - Pour maîtriser les Transformers
   - Directement applicable à ton projet

### 📄 Papers Importants

1. **"Attention Is All You Need" (2017)**
   - Le paper original des Transformers
   - Indispensable pour comprendre l'architecture

2. **"Music Transformer" (2018)**
   - Adaptation des Transformers à la musique
   - Base de notre projet

3. **"MuseNet" by OpenAI**
   - Génération musicale multi-instruments
   - Idées pour étendre le projet

### 🛠️ Outils et Frameworks

1. **Weights & Biases**
   ```python
   import wandb
   wandb.init(project="midi-ai")
   wandb.log({"loss": loss, "epoch": epoch})
   ```
   Tracking d'expériences ML professionnels

2. **Ray Tune**
   ```python
   from ray import tune
   tune.run(train_model, config={
       "lr": tune.grid_search([1e-4, 1e-3]),
       "batch_size": tune.choice([16, 32, 64])
   })
   ```
   Hyperparameter tuning automatique

3. **TensorBoard**
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter()
   writer.add_scalar('Loss/train', loss, epoch)
   ```

### 🎯 Projets pour Pratiquer

1. **Classifier des Genres Musicaux**
   - Plus simple que la génération
   - Apprends les CNN sur spectrogrammes

2. **Drum Pattern Generator**
   - Focus sur le rythme uniquement
   - Plus simple à évaluer

3. **Chord Progression Suggester**
   - Prédit le prochain accord
   - Application pratique pour musiciens

### 💡 Conseils de Progression

1. **Commence Petit**
   - Modifie d'abord les hyperparamètres
   - Observe les effets sur la génération

2. **Expérimente**
   - Change la température, observe
   - Modifie l'architecture, compare
   - Essaie différents styles de tokens

3. **Documente**
   - Tiens un journal de tes expériences
   - Note ce qui marche/ne marche pas

4. **Partage**
   - Publie tes améliorations sur GitHub
   - Écris des articles Medium sur tes découvertes
   - Contribue à des projets open source

### 🏆 Pour ton CV

Après ce projet, tu peux légitimement mettre :
- **"Développement de modèles Deep Learning"**
- **"Implémentation d'architectures Transformer"**
- **"Optimisation GPU sur Apple Silicon"**
- **"Traitement de données séquentielles"**

Prochaines étapes pour consolider :
1. Implémenter une variation (ex: GAN pour MIDI)
2. Publier un package npm qui interface avec ton modèle Python
3. Créer une API REST pour la génération
4. Déployer sur Hugging Face Spaces

---

## 🎬 Conclusion

Ce projet t'a fait toucher à :
- **Preprocessing** de données complexes (MIDI → tokens)
- **Architecture** de modèles state-of-the-art (Transformers)
- **Entraînement** optimisé hardware (MPS)
- **Génération** créative avec contrôle fin

Tu as maintenant les bases pour :
- Comprendre les papers de recherche
- Implémenter tes propres architectures
- Optimiser pour ton hardware
- Créer des projets IA créatifs

**Remember** : L'IA c'est 20% de modèles et 80% d'ingénierie. Ton background en développement est un atout énorme !

Bon voyage dans le monde du Deep Learning ! 🚀

---

*"The best way to learn AI is to build something creative with it."*