#!/usr/bin/env python3
"""
MIDI Tokenizer - Conversion MIDI vers tokens pour l'entraînement
Utilise REMI+ (REpresentation of Music with Improvements)
"""

import os
import numpy as np
import torch
from miditok import REMI, TokenizerConfig
import pretty_midi
from tqdm import tqdm
import pickle
import json

class MidiTokenizer:
    def __init__(self, 
                 vocab_size=512,
                 beat_res={(0, 4): 8, (4, 12): 4},
                 nb_velocities=32,
                 max_bar_length=256):
        
        # Configuration du tokenizer REMI
        config = TokenizerConfig(
            num_velocities=nb_velocities,
            beat_res=beat_res,
            use_chords=True,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=False,  # On se concentre sur un seul instrument
            nb_tempos=32,
            tempo_range=(60, 200),
            time_signature_range={4: [4]}  # Signature 4/4 par défaut
        )
        
        self.tokenizer = REMI(config)
        self.max_bar_length = max_bar_length
        
    def process_dataset(self, midi_dir="./midi_files", output_dir="./tokenized_data"):
        """Tokenize tous les fichiers MIDI"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Collecter tous les fichiers MIDI
        midi_files = []
        for root, dirs, files in os.walk(midi_dir):
            for file in files:
                if file.endswith('.mid'):
                    midi_files.append(os.path.join(root, file))
        
        print(f"Tokenization de {len(midi_files)} fichiers...")
        
        # Tokenizer chaque fichier
        all_tokens = []
        metadata = []
        
        for filepath in tqdm(midi_files, desc="Tokenization"):
            try:
                # Tokenize le fichier
                tokens = self.tokenizer(filepath)
                
                # Découper en segments de 16 secondes (~64 bars à 120 BPM)
                segments = self._segment_tokens(tokens, bars_per_segment=64)
                
                for i, segment in enumerate(segments):
                    all_tokens.append(segment)
                    metadata.append({
                        'source_file': os.path.basename(filepath),
                        'segment_idx': i,
                        'length': len(segment)
                    })
                    
            except Exception as e:
                print(f"Erreur avec {filepath}: {e}")
                continue
        
        # Sauvegarder les données
        print(f"\n💾 Sauvegarde de {len(all_tokens)} segments...")
        
        # Sauvegarder les tokens
        with open(os.path.join(output_dir, 'tokens.pkl'), 'wb') as f:
            pickle.dump(all_tokens, f)
            
        # Sauvegarder les métadonnées
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Sauvegarder le vocabulaire
        vocab = self.tokenizer.vocab
        with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
            json.dump({
                'vocab_size': len(vocab),
                'special_tokens': self.tokenizer.special_tokens,
                'vocab_items': list(vocab.items())[:100]  # Aperçu
            }, f, indent=2)
        
        print(f"✅ Tokenization terminée!")
        print(f"- Segments créés: {len(all_tokens)}")
        print(f"- Taille du vocabulaire: {len(vocab)}")
        print(f"- Longueur moyenne: {np.mean([len(t) for t in all_tokens]):.1f} tokens")
        
        return all_tokens, metadata
    
    def _segment_tokens(self, tokens, bars_per_segment=64):
        """Découpe les tokens en segments de taille fixe"""
        segments = []
        current_segment = []
        bar_count = 0
        
        for token in tokens[0]:  # tokens est une liste de listes
            current_segment.append(token)
            
            # Détecter les barres
            if 'Bar' in str(token):
                bar_count += 1
                
                if bar_count >= bars_per_segment:
                    segments.append(current_segment)
                    current_segment = []
                    bar_count = 0
        
        # Ajouter le dernier segment s'il n'est pas vide
        if len(current_segment) > 10:  # Au moins 10 tokens
            segments.append(current_segment)
            
        return segments
    
    def create_dataset(self, tokenized_dir="./tokenized_data"):
        """Créer un dataset PyTorch"""
        # Charger les tokens
        with open(os.path.join(tokenized_dir, 'tokens.pkl'), 'rb') as f:
            all_tokens = pickle.load(f)
        
        # Convertir en tenseurs PyTorch
        dataset = []
        for tokens in all_tokens:
            # Convertir les tokens en indices
            token_ids = [self.tokenizer.vocab.get(str(t), 0) for t in tokens]
            dataset.append(torch.tensor(token_ids, dtype=torch.long))
        
        return dataset

if __name__ == "__main__":
    # Test du tokenizer
    tokenizer = MidiTokenizer()
    tokens, metadata = tokenizer.process_dataset()