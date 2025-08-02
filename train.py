#!/usr/bin/env python3
"""
Script d'entraînement optimisé pour Apple Silicon (MPS)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import json
import pickle
from datetime import datetime
# import wandb  # Optionnel
from model import MusicTransformer
from midi_tokenizer import MidiTokenizer

class MidiDataset(Dataset):
    """Dataset pour les séquences MIDI tokenisées"""
    def __init__(self, tokenized_dir="./tokenized_data", max_seq_len=512):
        # Charger les tokens
        with open(os.path.join(tokenized_dir, 'tokens.pkl'), 'rb') as f:
            self.sequences = pickle.load(f)
            
        self.max_seq_len = max_seq_len
        print(f"Dataset chargé: {len(self.sequences)} séquences")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Tronquer ou padder la séquence
        if len(seq) > self.max_seq_len:
            # Prendre un segment aléatoire
            start = np.random.randint(0, len(seq) - self.max_seq_len)
            seq = seq[start:start + self.max_seq_len]
        else:
            # Padder avec des zéros
            seq = seq + [0] * (self.max_seq_len - len(seq))
        
        # Convertir en tenseur
        seq = torch.tensor(seq, dtype=torch.long)
        
        # Input: tous sauf le dernier, Target: tous sauf le premier
        return seq[:-1], seq[1:]

class Trainer:
    def __init__(self, model_config, training_config):
        self.device = self._setup_device()
        self.model_config = model_config
        self.training_config = training_config
        
        # Créer le modèle
        self.model = MusicTransformer(**model_config).to(self.device)
        print(f"Modèle créé avec {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M paramètres")
        
        # Optimizer et scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorer le padding
        
        # Dossier de sauvegarde
        self.checkpoint_dir = f"./checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def _setup_device(self):
        """Configuration du device avec support MPS"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🚀 Utilisation du GPU Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("🚀 Utilisation de CUDA")
        else:
            device = torch.device("cpu")
            print("💻 Utilisation du CPU")
        return device
    
    def train(self, train_loader, val_loader=None, epochs=50):
        """Boucle d'entraînement principale"""
        print(f"\n🎯 Début de l'entraînement pour {epochs} epochs")
        
        # Scheduler
        scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs * len(train_loader),
            eta_min=1e-6
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (x, y) in enumerate(train_bar):
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(x)
                
                # Calculer la loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                scheduler.step()
                
                # Stats
                train_loss += loss.item()
                train_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })
                
                # Log to wandb (si activé)
                # if batch_idx % 10 == 0 and self.training_config.get('use_wandb', False):
                #     wandb.log({
                #         'train_loss': loss.item(),
                #         'learning_rate': scheduler.get_last_lr()[0],
                #         'epoch': epoch
                #     })
            
            # Validation
            avg_train_loss = train_loss / len(train_loader)
            val_loss = self.validate(val_loader) if val_loader else avg_train_loss
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Sauvegarder le meilleur modèle
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            # Sauvegarder régulièrement
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss)
                
    def validate(self, val_loader):
        """Validation du modèle"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )
                val_loss += loss.item()
                
        return val_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Sauvegarde du checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch+1}.pt'
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint sauvegardé: {filepath}")

def main():
    # Configuration du modèle
    model_config = {
        'vocab_size': 512,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 512,
        'dropout': 0.1
    }
    
    # Configuration de l'entraînement
    training_config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 100,
        'use_wandb': False  # Mettre True pour utiliser Weights & Biases
    }
    
    # Créer le dataset
    print("📊 Chargement du dataset...")
    dataset = MidiDataset(max_seq_len=model_config['max_seq_len'])
    
    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Entraîner
    trainer = Trainer(model_config, training_config)
    trainer.train(train_loader, val_loader, epochs=training_config['epochs'])

if __name__ == "__main__":
    # Vérifier que les données sont tokenisées
    if not os.path.exists('./tokenized_data/tokens.pkl'):
        print("⚠️  Données non tokenisées. Exécutez d'abord: python midi_tokenizer.py")
    else:
        main()