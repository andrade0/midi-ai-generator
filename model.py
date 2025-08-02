#!/usr/bin/env python3
"""
Music Transformer - Architecture optimisée pour génération MIDI sur Apple Silicon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class RelativePositionalEncoding(nn.Module):
    """Encodage positionnel relatif pour capturer les patterns musicaux"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Créer la matrice d'encodage
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MusicTransformerBlock(nn.Module):
    """Block Transformer avec attention causale pour la musique"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention avec connexion résiduelle
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward avec connexion résiduelle
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class MusicTransformer(nn.Module):
    """
    Transformer pour génération musicale
    Optimisé pour Apple Silicon MPS
    """
    def __init__(self, 
                 vocab_size=512,
                 d_model=512,
                 n_heads=8,
                 n_layers=6,
                 d_ff=2048,
                 max_seq_len=2048,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = RelativePositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MusicTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Sortie
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialisation
        self._init_weights()
        
    def _init_weights(self):
        """Initialisation Xavier/He pour meilleure convergence"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def generate_square_subsequent_mask(self, size):
        """Masque causal pour l'attention"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Créer le masque causal si nécessaire
        if mask is None:
            device = x.device
            mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        # Embeddings + position
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Passer par tous les blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Sortie
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, 
                 start_tokens, 
                 max_length=512,
                 temperature=1.0,
                 top_k=50,
                 top_p=0.95):
        """
        Génération de séquence avec sampling
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Convertir en tenseur si nécessaire
        if isinstance(start_tokens, list):
            tokens = torch.tensor(start_tokens).unsqueeze(0).to(device)
        else:
            tokens = start_tokens
            
        for _ in range(max_length - tokens.shape[1]):
            # Limiter à la longueur max du modèle
            input_tokens = tokens[:, -self.max_seq_len:]
            
            # Forward pass
            logits = self(input_tokens)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Enlever les tokens avec probabilité cumulative > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Échantillonner
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Ajouter à la séquence
            tokens = torch.cat([tokens, next_token], dim=-1)
            
            # Arrêter si on génère un token de fin
            if next_token.item() == 0:  # Assuming 0 is EOS
                break
                
        return tokens

class MusicTransformerLightning(nn.Module):
    """Version Lightning pour entraînement optimisé"""
    def __init__(self, model_config):
        super().__init__()
        self.model = MusicTransformer(**model_config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            betas=(0.9, 0.98),
            eps=1e-9
        )