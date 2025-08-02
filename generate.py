#!/usr/bin/env python3
"""
Générateur de patterns MIDI à partir du modèle entraîné
"""

import os
import torch
import numpy as np
from datetime import datetime
import pretty_midi
from miditok import REMI, TokenizerConfig
from model import MusicTransformer
from midi_tokenizer import MidiTokenizer
import json
import argparse

class MidiGenerator:
    def __init__(self, checkpoint_path, device=None):
        # Charger le checkpoint
        print(f"📦 Chargement du modèle depuis {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Setup device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🚀 Utilisation du GPU Apple Silicon (MPS)")
            else:
                self.device = torch.device("cpu")
                print("💻 Utilisation du CPU")
        else:
            self.device = device
            
        # Recréer le modèle
        self.model = MusicTransformer(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Recréer le tokenizer
        self.setup_tokenizer()
        
        # Créer le dossier de sortie
        os.makedirs('./output_midi', exist_ok=True)
        
    def setup_tokenizer(self):
        """Configure le tokenizer REMI"""
        config = TokenizerConfig(
            num_velocities=32,
            beat_res={(0, 4): 8, (4, 12): 4},
            use_chords=True,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=False,
            nb_tempos=32,
            tempo_range=(60, 200),
            time_signature_range={4: [4]}  # 4/4 par défaut
        )
        self.tokenizer = REMI(config)
        
    def generate_midi_clip(self, 
                          style="random",
                          temperature=1.0,
                          top_k=50,
                          top_p=0.95,
                          length=512,
                          tempo=120):
        """
        Génère un pattern MIDI de 16 secondes
        
        Args:
            style: Style de départ ("chord", "melody", "bass", "random")
            temperature: Contrôle la créativité (0.5-1.5)
            top_k: Limite les choix aux K tokens les plus probables
            top_p: Nucleus sampling
            length: Longueur en tokens
            tempo: BPM cible
        """
        print(f"\n🎼 Génération d'un pattern {style} à {tempo} BPM...")
        
        # Créer les tokens de départ selon le style
        start_tokens = self._get_start_tokens(style, tempo)
        
        # Générer la séquence
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_length=length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Convertir les tokens en MIDI
        tokens = generated[0].cpu().numpy().tolist()
        
        # Décoder en MIDI
        midi_data = self._tokens_to_midi(tokens, tempo)
        
        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{style}_{tempo}bpm_{timestamp}.mid"
        filepath = os.path.join('./output_midi', filename)
        
        midi_data.write(filepath)
        print(f"✅ Pattern généré: {filepath}")
        
        # Retourner les infos
        return {
            'filepath': filepath,
            'duration': midi_data.get_end_time(),
            'n_notes': sum(len(inst.notes) for inst in midi_data.instruments),
            'tempo': tempo
        }
    
    def _get_start_tokens(self, style, tempo):
        """Crée les tokens de départ selon le style"""
        # Tokens de base: Tempo, Time Signature, Bar
        base_tokens = [
            f"Tempo_{tempo}",
            "TimeSig_4/4",
            "Bar_None"
        ]
        
        # Ajouter des tokens spécifiques au style
        if style == "chord":
            base_tokens.extend([
                "Position_0",
                "Chord_C:maj",
                "Position_96",  # 1 beat
                "Chord_Am:min"
            ])
        elif style == "melody":
            base_tokens.extend([
                "Position_0",
                "Pitch_60",  # C4
                "Velocity_80",
                "Duration_48"  # Half beat
            ])
        elif style == "bass":
            base_tokens.extend([
                "Position_0", 
                "Pitch_36",  # C2
                "Velocity_100",
                "Duration_96"  # 1 beat
            ])
        else:  # random
            base_tokens.extend([
                "Position_0",
                "Pitch_60",
                "Velocity_80",
                "Duration_48"
            ])
        
        # Convertir en indices
        token_ids = []
        for token in base_tokens:
            if token in self.tokenizer.vocab:
                token_ids.append(self.tokenizer.vocab[token])
            else:
                # Fallback sur un token similaire ou 0
                token_ids.append(0)
                
        return torch.tensor([token_ids], device=self.device)
    
    def _tokens_to_midi(self, tokens, tempo):
        """Convertit les tokens en objet MIDI"""
        # Créer un objet MIDI
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Créer un instrument (piano par défaut)
        instrument = pretty_midi.Instrument(program=0)
        
        # Variables d'état
        current_position = 0
        current_velocity = 80
        note_starts = {}
        
        # Parser les tokens
        for token in tokens:
            if isinstance(token, int):
                # Convertir l'index en nom de token
                token_name = self._idx_to_token(token)
            else:
                token_name = str(token)
                
            # Parser le token
            if token_name.startswith("Bar"):
                current_position += 96  # 1 bar = 96 ticks
                
            elif token_name.startswith("Position_"):
                current_position = int(token_name.split("_")[1])
                
            elif token_name.startswith("Pitch_"):
                pitch = int(token_name.split("_")[1])
                if pitch in note_starts:
                    # Finir la note précédente
                    start_time = note_starts[pitch]
                    end_time = current_position / 96 * (60 / tempo) * 4
                    note = pretty_midi.Note(
                        velocity=current_velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)
                    del note_starts[pitch]
                    
                # Commencer une nouvelle note
                note_starts[pitch] = current_position / 96 * (60 / tempo) * 4
                
            elif token_name.startswith("Velocity_"):
                current_velocity = int(token_name.split("_")[1])
                
            elif token_name.startswith("Duration_"):
                # Pour les tokens avec durée explicite
                duration_ticks = int(token_name.split("_")[1])
                if note_starts:
                    last_pitch = list(note_starts.keys())[-1]
                    start_time = note_starts[last_pitch]
                    end_time = start_time + duration_ticks / 96 * (60 / tempo) * 4
                    note = pretty_midi.Note(
                        velocity=current_velocity,
                        pitch=last_pitch,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)
                    del note_starts[last_pitch]
        
        # Finir toutes les notes restantes
        for pitch, start_time in note_starts.items():
            end_time = current_position / 96 * (60 / tempo) * 4
            note = pretty_midi.Note(
                velocity=current_velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
            
        # Ajouter l'instrument au MIDI
        midi.instruments.append(instrument)
        
        # S'assurer que la durée est d'environ 16 secondes
        if midi.get_end_time() < 16:
            # Répéter le pattern si nécessaire
            original_notes = instrument.notes.copy()
            while midi.get_end_time() < 16:
                offset = midi.get_end_time()
                for note in original_notes:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + offset,
                        end=note.end + offset
                    )
                    instrument.notes.append(new_note)
                    
        return midi
    
    def _idx_to_token(self, idx):
        """Convertit un index en nom de token"""
        # Inverser le vocabulaire
        if not hasattr(self, 'idx_to_vocab'):
            self.idx_to_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return self.idx_to_vocab.get(idx, "Unknown")
    
    def batch_generate(self, n_clips=10, styles=None, tempo_range=(110, 130)):
        """Génère plusieurs clips avec des paramètres variés"""
        if styles is None:
            styles = ["chord", "melody", "bass", "random"]
            
        results = []
        
        for i in range(n_clips):
            style = np.random.choice(styles)
            tempo = np.random.randint(tempo_range[0], tempo_range[1])
            temperature = np.random.uniform(0.7, 1.2)
            
            result = self.generate_midi_clip(
                style=style,
                temperature=temperature,
                tempo=tempo
            )
            results.append(result)
            
        print(f"\n🎹 {n_clips} patterns générés dans ./output_midi/")
        return results

def main():
    parser = argparse.ArgumentParser(description="Générateur de patterns MIDI")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--style', type=str, default='random',
                       choices=['chord', 'melody', 'bass', 'random'],
                       help='Style de génération')
    parser.add_argument('--tempo', type=int, default=120,
                       help='Tempo en BPM')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Température de sampling (0.5-1.5)')
    parser.add_argument('--n_clips', type=int, default=1,
                       help='Nombre de clips à générer')
    
    args = parser.parse_args()
    
    # Vérifier que le checkpoint existe
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint non trouvé: {args.checkpoint}")
        print("Entraînez d'abord le modèle avec: python train.py")
        return
        
    # Créer le générateur
    generator = MidiGenerator(args.checkpoint)
    
    # Générer
    if args.n_clips == 1:
        generator.generate_midi_clip(
            style=args.style,
            temperature=args.temperature,
            tempo=args.tempo
        )
    else:
        generator.batch_generate(n_clips=args.n_clips)

if __name__ == "__main__":
    main()