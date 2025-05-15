"""Postprocessing functions for Whisper-generated transcriptions."""

import numpy as np
import re


excluded_tokens = [11, 13]  # Punctuation tokens to exclude from repetition penalty

def apply_repetition_penalty(logits, generated_tokens, penalty=1.5, last_window=8):
    """
    Apply repetition penalty to the logits.
    Args:
        logits: The logits from the model (shape: (vocab_size,)).
        generated_tokens: List of previously generated tokens.
        penalty: The penalty factor (higher values discourage repetitions).
    Returns:
        logits: The logits with repetition penalty applied.
    """
    logits = np.squeeze(logits, axis=0)
    recent_tokens = generated_tokens[-last_window:] if len(generated_tokens) >= last_window else generated_tokens

    # Combine the tokens from both windows
    recent_tokens = set(recent_tokens)

    for token in recent_tokens:
        if token not in excluded_tokens:
            logits[token] /= penalty
    return logits

def temperature_sampling(logits, temperature=0.0):
    """
    Apply temperature sampling to the logits.
    """
    # Boost the logits for punctuation tokens
    for punct_idx in excluded_tokens:
        if punct_idx < len(logits):
            logits[punct_idx] *= 1.2

    if temperature == 0.0:
        return np.argmax(logits)  # Greedy decoding
    # Subtract max for numerical stability
    logits = logits - np.max(logits)
    logits = logits / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    if np.isnan(probs).any():
        print("Warning: Probabilities contain NaN values. Falling back to greedy decoding.")
        return np.argmax(logits)  # Fall back to greedy decoding
    # Ensure probabilities sum to 1
    probs = probs / np.sum(probs)
    next_token = np.random.choice(len(probs), p=probs)  # Sample from the distribution
    return next_token


def clean_transcription(transcription):
    # Split the transcription into sentences using both '.' and '?' as delimiters
    sentences = re.split(r'(?<=[.?])\s+', transcription)
    
    # Initialize a list to store unique sentences
    unique_sentences = []
    
    # Iterate through the sentences
    for sentence in sentences:
        # Check if any part of the current sentence has already appeared in the unique sentences
        for unique_sentence in unique_sentences:
            # Normalize both sentences for comparison
            normalized_current = sentence.lower().strip()
            normalized_unique = unique_sentence.lower().strip()
            
            # Check if the current sentence is a substring of the unique sentence or vice versa
            if normalized_current in normalized_unique or normalized_unique in normalized_current:
                # If a repetition is found, stop processing and return the cleaned transcription
                cleaned_transcription = ' '.join(unique_sentences)
                #cleaned_transcription = '. '.join(unique_sentences)
                # Ensure the last character is a proper delimiter (e.g., '.' or '?')
                if not cleaned_transcription.endswith(('.', '?')):
                    cleaned_transcription += '.'
                return cleaned_transcription
        
        # If no repetition is found, add the current sentence to the unique list
        unique_sentences.append(sentence.strip())
    
    # If no repetition is found, join all sentences and return
    #cleaned_transcription = '. '.join(unique_sentences)
    cleaned_transcription = ' '.join(unique_sentences)
    # Ensure the last character is a proper delimiter (e.g., '.' or '?')
    if not cleaned_transcription.endswith(('.', '?')):
        cleaned_transcription += '.'
    return cleaned_transcription
