# ml_projec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time
import random

# ========== SETTINGS ==========
file_paths = [
    "/kaggle/input/frisbee/general_tips_pretraining.txt",
    "/kaggle/input/frisbee/frisbee_reddit_qna.txt",
    "/kaggle/input/frisbee/frisbee_coaching_data.txt"
]

stage_names = ["General Pretraining", "Frisbee Knowledge Fill", "Coaching Fine-tune"]
stage_iters = [1000, 1250, 1500]  # Customize training steps for each stage

# Enhanced model configuration
n_embed = 256      # Increased from 192
n_head = 8         # Increased from 6
n_layer = 4        # Increased from 3
block_size = 256   # Context length
batch_size = 32    # Batch size
learning_rate = 1e-3
dropout_rate = 0.2 # Keep the same
eval_interval = 250
eval_iters = 50

# ========== DEVICE ==========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
torch.manual_seed(42)

# ========== TOKENIZATION ==========
def build_combined_vocab(file_paths):
    """Build a combined vocabulary from all datasets"""
    all_text = ""
    for path in file_paths:
        text = Path(path).read_text(encoding='utf-8')
        all_text += text + "\n"
    
    chars = sorted(list(set(all_text)))
    # Add special tokens if not present
    special_tokens = ['?', '!', '.', ',']
    for token in special_tokens:
        if token not in chars:
            chars.append(token)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos

# ========== DATA UTILS ==========
def encode_safe(text, stoi, unk_token='?'):
    """Safely encode text, handling unknown characters"""
    encoded = []
    for c in text:
        if c in stoi:
            encoded.append(stoi[c])
        elif unk_token in stoi:
            # Replace unknown characters with the unknown token
            encoded.append(stoi[unk_token])
        else:
            # Skip unknown characters if no unknown token is provided
            pass
    return encoded

def decode(indices, itos):
    """Convert indices back to text"""
    return ''.join([itos[i] for i in indices])

def get_batch(data, split, block_size, batch_size):
    """Get a random batch of data"""
    ix = torch.randint(len(data[split]) - block_size, (batch_size,))
    x = torch.stack([data[split][i:i+block_size] for i in ix])
    y = torch.stack([data[split][i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size):
    """Estimate loss on train and validation splits"""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, split, block_size, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ========== DATA ENHANCEMENT FUNCTIONS ==========
def augment_data(text):
    """Simple data augmentation by creating variations of the text"""
    lines = text.strip().split('\n')
    augmented_lines = []
    
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            input_line = lines[i]
            output_line = lines[i+1]
            
            # Add original pair
            augmented_lines.append(input_line)
            augmented_lines.append(output_line)
            
            # Create variations by swapping words or adding synonyms
            if "INPUT:" in input_line and "OUTPUT:" in output_line:
                # Extract the question part
                question = input_line.replace("INPUT:", "").strip()
                answer = output_line.replace("OUTPUT:", "").strip()
                
                # Create variations with different question formulations
                variations = [
                    f"INPUT: How can I {question.lower().replace('how do i ', '')}",
                    f"INPUT: What's the best way to {question.lower().replace('how do i ', '').replace('what is the best way to ', '')}"
                ]
                
                for var in variations:
                    augmented_lines.append(var)
                    augmented_lines.append(output_line)
    
    return '\n'.join(augmented_lines)

def sort_by_complexity(text):
    """Sort training examples by complexity (length)"""
    lines = text.strip().split('\n')
    examples = []
    
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            input_line = lines[i]
            output_line = lines[i+1]
            # Use length as a simple proxy for complexity
            complexity = len(input_line) + len(output_line)
            examples.append((input_line, output_line, complexity))
    
    # Sort by complexity
    examples.sort(key=lambda x: x[2])
    
    # Reconstruct text
    sorted_lines = []
    for input_line, output_line, _ in examples:
        sorted_lines.append(input_line)
        sorted_lines.append(output_line)
    
    return '\n'.join(sorted_lines)

def organize_by_topic(file_paths):
    """Organize training data by topics"""
    topics = {
        "throwing": ["backhand", "forehand", "throw", "huck", "hammer", "scoober"],
        "defense": ["defend", "mark", "force", "cover", "block"],
        "cutting": ["cut", "get open", "fake", "juke"],
        "strategy": ["stack", "play", "offense", "defense", "zone", "man"],
        "mental": ["focus", "visualize", "calm", "pressure", "timeout"]
    }
    
    organized_data = {topic: [] for topic in topics}
    general_data = []
    
    for path in file_paths:
        text = Path(path).read_text(encoding='utf-8')
        lines = text.strip().split('\n')
        
        for i in range(0, len(lines), 2):
            if i+1 < len(lines):
                input_line = lines[i]
                output_line = lines[i+1]
                
                # Check which topic this belongs to
                assigned = False
                for topic, keywords in topics.items():
                    if any(keyword in input_line.lower() for keyword in keywords):
                        organized_data[topic].append(input_line)
                        organized_data[topic].append(output_line)
                        assigned = True
                        break
                
                if not assigned:
                    general_data.append(input_line)
                    general_data.append(output_line)
    
    # Combine all data back together, but now organized by topic
    combined_data = []
    for topic, data in organized_data.items():
        combined_data.extend(data)
    combined_data.extend(general_data)
    
    return '\n'.join(combined_data)

def apply_response_templates(text):
    """Apply response templates to ensure consistent output format"""
    lines = text.strip().split('\n')
    templated_lines = []
    
    templates = [
        "To {action}, you should {advice}.",
        "{advice} is key for {action}.",
        "When {action}, remember to {advice}.",
        "The best way to {action} is to {advice}."
    ]
    
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            input_line = lines[i]
            output_line = lines[i+1]
            
            if "INPUT:" in input_line and "OUTPUT:" in output_line:
                # Extract the question and answer
                question = input_line.replace("INPUT:", "").strip()
                answer = output_line.replace("OUTPUT:", "").strip()
                
                # Try to extract action from question
                action = question.lower()
                for prefix in ["how do i ", "how to ", "what's the best way to "]:
                    if action.startswith(prefix):
                        action = action.replace(prefix, "")
                        break
                
                # If answer is too short, try to expand it using templates
                if len(answer) < 20:
                    template = random.choice(templates)
                    new_answer = template.format(action=action, advice=answer)
                    output_line = f"OUTPUT: {new_answer}"
            
            templated_lines.append(input_line)
            templated_lines.append(output_line)
    
    return '\n'.join(templated_lines)

# ========== POSITIONAL ENCODING ==========
class RotaryPositionalEncoding(nn.Module):
    def _init_(self, dim, max_seq_len=5000):
        super()._init_()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        seq_len = x.shape[1] if seq_len is None else seq_len
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]
        
    def apply_rotary_pos_emb(self, x, freqs):
        # Implement the rotation for the embeddings
        x_complex = torch.complex(x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:])
        freqs_complex = torch.complex(freqs[..., :freqs.shape[-1]//2], freqs[..., freqs.shape[-1]//2:])
        x_rotated = x_complex * freqs_complex
        x_out = torch.cat([x_rotated.real, x_rotated.imag], dim=-1)
        return x_out

# Standard positional encoding as fallback
class PositionalEncoding(nn.Module):
    def _init_(self, d_model, max_len=5000):
        super()._init_()
        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1), :]

# ========== ATTENTION MECHANISM ==========
class SelfAttentionHead(nn.Module):
    def _init_(self, head_size):
        super()._init_()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Apply attention to values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def _init_(self, num_heads, head_size):
        super()._init_()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def _init_(self):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def _init_(self):
        super()._init_()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Pre-LN architecture for better training stability
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def _init_(self, vocab_size):
        super()._init_()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_encoding = PositionalEncoding(n_embed, block_size)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = self.position_encoding(tok_emb)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_view = logits.view(B * T, -1)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)
        return logits, loss

    def generate(self, idx, max_new_tokens, itos, temperature=0.7, top_k=50, top_p=0.9):
        """Generate text with top-k and top-p sampling for better quality"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if it's too long
                idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
                
                # Get predictions
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we generate a newline after OUTPUT:
                if decode(next_token[0].tolist(), itos) == '\n' and 'OUTPUT:' in decode(idx[0].tolist(), itos):
                    break
                
                # Add to context
                idx = torch.cat((idx, next_token), dim=1)
        
        return idx

# ========== ADVANCED GENERATION METHODS ==========
def generate_with_beam_search(model, prompt, stoi, itos, beam_width=5, max_tokens=100):
    """Generate text using beam search for more coherent outputs"""
    context = torch.tensor([encode_safe(prompt, stoi)], dtype=torch.long).to(device)
    
    # Initialize with the input sequence
    sequences = [(context, 0.0)]  # (sequence, score)
    
    for _ in range(max_tokens):
        all_candidates = []
        
        # Expand each current candidate
        for seq, score in sequences:
            # Get the next token probabilities
            with torch.no_grad():
                idx_cond = seq[:, -block_size:] if seq.size(1) > block_size else seq
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
            
            # Get top-k next tokens and their probabilities
            top_k_probs, top_k_indices = torch.topk(probs, beam_width)
            
            # Create new candidates
            for i in range(beam_width):
                next_token = top_k_indices[:, i:i+1]
                next_score = score + torch.log(top_k_probs[:, i]).item()
                next_seq = torch.cat((seq, next_token), dim=1)
                
                # Check if this is a complete sequence
                token_str = decode(next_token[0].tolist(), itos)
                if token_str == '\n' and 'OUTPUT:' in decode(seq[0].tolist(), itos):
                    all_candidates.append((next_seq, next_score, True))  # Mark as complete
                else:
                    all_candidates.append((next_seq, next_score, False))
        
        # Select top-k candidates, prioritizing complete sequences
        complete_candidates = [c for c in all_candidates if c[2]]
        incomplete_candidates = [c for c in all_candidates if not c[2]]
        
        # If we have enough complete candidates, use those
        if len(complete_candidates) >= beam_width:
            ordered = sorted(complete_candidates, key=lambda x: x[1], reverse=True)
            sequences = [(seq, score) for seq, score, _ in ordered[:beam_width]]
            break
        else:
            # Otherwise, use the best incomplete candidates
            ordered = sorted(incomplete_candidates, key=lambda x: x[1], reverse=True)
            sequences = [(seq, score) for seq, score, _ in ordered[:beam_width]]
    
    # Return the highest-scoring sequence
    return sequences[0][0]

def optimize_temperature(model, prompt, stoi, itos, temperatures=[0.5, 0.7, 0.9, 1.1]):
    """Find the optimal temperature for a given prompt"""
    results = []
    
    for temp in temperatures:
        response = generate_text(model, prompt, stoi, itos, max_tokens=100, temperature=temp)
        try:
            output_text = response.split("OUTPUT:")[1].strip()
            # Calculate a simple quality score based on length and relevance
            score = len(output_text) / 10  # Longer responses get higher scores
            # Check for relevance by looking for keywords from the prompt in the response
            keywords = [word for word in prompt.lower().split() if len(word) > 3]
            for keyword in keywords:
                if keyword in output_text.lower():
                    score += 2  # Bonus for each relevant keyword
            results.append((temp, output_text, score))
        except IndexError:
            results.append((temp, "[Failed to generate]", 0))
    
    # Return the response with the highest score
    results.sort(key=lambda x: x[2], reverse=True)
    return results[0][0], results[0][1]  # Return best temperature and response

# ========== EVALUATION METRICS ==========
def evaluate_model_quality(model, stoi, itos):
    """Evaluate model quality with multiple metrics"""
    test_prompts = [
        "How do I improve my backhand?",
        "What's the best way to mark a thrower?",
        "Tips for better cutting?",
        "How to defend against a huck?",
        "What should I do when I'm at stall 8?",
        "How can I practice throws alone?"
    ]
    
    # Expected keywords for each prompt
    expected_keywords = [
        ["backhand", "throw", "wrist", "grip", "follow"],
        ["mark", "force", "stance", "position", "thrower"],
        ["cut", "fake", "juke", "open", "space"],
        ["huck", "deep", "position", "defend", "track"],
        ["stall", "dump", "reset", "quick", "release"],
        ["practice", "throw", "target", "drill", "wall"]
    ]
    
    results = []
    total_relevance = 0
    total_coherence = 0
    
    for i, prompt in enumerate(test_prompts):
        full_prompt = f"INPUT: {prompt}\nOUTPUT:"
        
        # Find optimal temperature for this prompt
        best_temp, response = optimize_temperature(model, full_prompt, stoi, itos)
        
        # Calculate relevance score
        relevance = 0
        for keyword in expected_keywords[i]:
            if keyword in response.lower():
                relevance += 1
        relevance = relevance / len(expected_keywords[i])
        
        # Calculate coherence score (simple heuristic based on sentence structure)
        coherence = 1.0
        if len(response.split()) < 3:
            coherence = 0.3
        elif not any(word in response.lower() for word in ["the", "to", "and", "your"]):
            coherence = 0.7
        
        total_relevance += relevance
        total_coherence += coherence
        
        results.append({
            "prompt": prompt,
            "response": response,
            "temperature": best_temp,
            "relevance": relevance,
            "coherence": coherence
        })
    
    avg_relevance = total_relevance / len(test_prompts)
    avg_coherence = total_coherence / len(test_prompts)
    
    print(f"Average Relevance: {avg_relevance:.2f}")
    print(f"Average Coherence: {avg_coherence:.2f}")
    
    return results, avg_relevance, avg_coherence

# ========== TRAINING FUNCTIONS ==========
def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent exploding gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def train_model(file_paths, stoi, itos):
    """Train the model through all stages with a single enhanced file"""
    train_losses = []
    val_losses = []
    total_iters = sum(stage_iters)
    
    # Initialize model
    vocab_size = len(stoi)
    model = TransformerLanguageModel(vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=learning_rate/10)
    
    current_iter = 0
    start_time = time.time()
    
    # Load the enhanced data once
    text = Path(file_paths[0]).read_text(encoding='utf-8')
    encoded = torch.tensor(encode_safe(text, stoi), dtype=torch.long)
    n = int(0.9 * len(encoded))
    data = {
        'train': encoded[:n],
        'val': encoded[n:]
    }

    # Generate sample after each stage
    print(f"\n--- Sample Generation after Stage 0 ---")
    prompt = f"INPUT: {get_sample_prompt(0)}\nOUTPUT:"
    sample = generate_text(model, prompt, stoi, itos, max_tokens=50)
    print(sample)
    
    # Run through all stages with the same data
    for stage in range(len(stage_names)):
        print(f"\n=== Stage {stage+1}: {stage_names[stage]} ===")
        
        # Training loop for this stage
        for iter in range(stage_iters[stage]):
            if iter % eval_interval == 0:
                losses = estimate_loss(model, data, block_size, batch_size)
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                print(f"Step {current_iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Get batch and compute loss
            xb, yb = get_batch(data, 'train', block_size, batch_size)
            logits, loss = model(xb, yb)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model)  # Apply gradient clipping
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            current_iter += 1
        
        # Save model checkpoint after each stage
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': {'stoi': stoi, 'itos': itos}
        }, f'/kaggle/working/frisbee_model_stage_{stage+1}.pt')

        # Generate sample after each stage
        print(f"\n--- Sample Generation after Stage {stage+1} ---")
        prompt = f"INPUT: {get_sample_prompt(stage)}\nOUTPUT:"
        sample = generate_text(model, prompt, stoi, itos, max_tokens=50)
        print(sample)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return model, train_losses, val_losses

def get_sample_prompt(stage):
    """Get appropriate sample prompts for each stage"""
    prompts = [
        "How to improve focus?",              # Stage 1
        "What's the best way to defend?",     # Stage 2
        "Coaching tip for pivot."             # Stage 3
    ]
    return prompts[min(stage, len(prompts)-1)]

def generate_text(model, prompt, stoi, itos, max_tokens=100, temperature=0.7):
    """Generate text from a prompt with temperature control"""
    context = torch.tensor([encode_safe(prompt, stoi)], dtype=torch.long).to(device)
    generated = model.generate(context, max_tokens, itos, temperature=temperature, top_k=50, top_p=0.9)
    return decode(generated[0].tolist(), itos)

# ========== FRISBEE-SPECIFIC FUNCTIONS ==========
def is_frisbee_related(question):
    """Check if a question is related to frisbee"""
    frisbee_keywords = [
        "frisbee", "disc", "throw", "catch", "cut", "mark", "stack", "handler", 
        "cutter", "backhand", "forehand", "huck", "pull", "stall", "pivot", 
        "defense", "offense", "zone", "man", "layout", "bid", "ultimate"
    ]
    
    question = question.lower()
    return any(keyword in question for keyword in frisbee_keywords)

def get_template_response(question):
    """Return template response for common questions if available"""
    templates = {
        "stall mark": "To effectively stall mark, maintain a low athletic stance with bent knees, keep your arms wide, and stay at arm's length from the thrower. Mirror their movements, watch their eyes and shoulders for cues, and count clearly. Focus on forcing them in the direction your team's defense is set up for.",
        
        "throw far": "To throw farther, focus on generating power from your core and hips, not just your arm. Keep your wrist firm, follow through completely, and release the disc at a slight upward angle with enough spin. Practice with proper form before adding power.",
        
        "cut": "For better cutting, start from a good ready position with knees bent, explode into your cuts with a sharp change of direction, and commit fully to each cut. Make eye contact with handlers and time your cuts based on field position.",
        
        "pivot": "When pivoting, keep your non-pivot foot firmly planted, maintain low center of gravity, and use your core for balance. Look upfield before pivoting to identify options, and practice quick, decisive movements to create throwing angles.",
        
        "backhand": "To improve your backhand, focus on proper grip with your index finger along the rim, keep your wrist firm, and follow through in the direction you want the disc to travel. Practice stepping out at a 90-degree angle from your target for better angles.",
        
        "forehand": "For a better forehand (flick), keep your elbow close to your body, snap your wrist firmly, and follow through toward your target. Start with short distances focusing on spin and flat release before adding power."
    }
    
    question = question.lower()
    for key, response in templates.items():
        if key in question:
            return response
    
    return None  # No template found

def post_process_response(response):
    """Clean and enhance the model's response"""
    # Truncate at punctuation if response is cut off
    if not response.endswith((".", "!", "?")):
        last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
        if last_punct > len(response) * 0.5:  # Only truncate if we're not losing too much
            response = response[:last_punct+1]
    
    # Fix common issues
    response = response.replace(" ,", ",")
    response = response.replace(" .", ".")
    response = response.replace("  ", " ")
    
    # Ensure response starts with capital letter
    if response and response[0].islower():
        response = response[0].upper() + response[1:]
    
    return response

# ========== INTERACTIVE INTERFACE ==========
def interactive_coach_enhanced(model, stoi, itos):
    """Enhanced interactive interface with multiple improvements"""
    print("\n=== Enhanced Frisbee AI Coach ===")
    print("Welcome! I'm a specialized AI coach built from scratch using a transformer architecture.")
    print("I was trained on frisbee coaching data through a three-stage process:")
    print("  1. General pretraining on coaching principles")
    print("  2. Knowledge filling with frisbee-specific information")
    print("  3. Fine-tuning on specialized coaching techniques")
    print("\nAsk a question about frisbee or type 'quit' to exit")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            break
        
        # Check if question is frisbee-related
        if not is_frisbee_related(user_input):
            print("\nCoach says: I'm sorry, but I'm specifically trained on frisbee coaching. I'm not sure how to answer questions outside that domain. Could you ask me something about frisbee techniques, strategies, or training?")
            continue
        
        # Check for template response
        template_response = get_template_response(user_input)
        if template_response:
            print(f"\nCoach says: {template_response}")
            continue
        
        # Generate response with beam search
        prompt = f"INPUT: {user_input}\nOUTPUT:"
        context = torch.tensor([encode_safe(prompt, stoi)], dtype=torch.long).to(device)
        
        # Try beam search for better quality
        generated = generate_with_beam_search(model, prompt, stoi, itos, beam_width=5)
        response = decode(generated[0].tolist(), itos)
        
        # Extract and process the output
        try:
            output_text = response.split("OUTPUT:")[1].strip()
            output_text = post_process_response(output_text)
            
            # Check if response makes sense for the question
            if len(output_text.split()) < 3 or not any(word in output_text.lower() for word in user_input.lower().split() if len(word) > 3):
                print("\nCoach says: I'm not entirely sure about that aspect of frisbee. Could you rephrase or ask about a specific technique?")
            else:
                print(f"\nCoach says: {output_text}")
        except IndexError:
            print("\nCoach says: I'm not sure how to answer that specific frisbee question. Could you try rephrasing it?")

def evaluate_model(model, stoi, itos):
    """Evaluate the model on a set of test prompts"""
    test_prompts = [
        "How do I improve my backhand?",
        "What's the best way to mark a thrower?",
        "Tips for better cutting?",
        "How to defend against a huck?",
        "What should I do when I'm at stall 8?",
        "How can I practice throws alone?"
    ]
    
    print("\n=== Model Evaluation ===")
    for prompt in test_prompts:
        full_prompt = f"INPUT: {prompt}\nOUTPUT:"
        response = generate_text(model, full_prompt, stoi, itos, max_tokens=100, temperature=0.7)
        
        try:
            output_text = response.split("OUTPUT:")[1].strip()
            print(f"\nQ: {prompt}")
            print(f"A: {output_text}")
            print("-" * 50)
        except IndexError:
            print(f"\nQ: {prompt}")
            print("A: [Failed to generate response]")
            print("-" * 50)

def main():
    # Build combined vocabulary from all datasets
    print("Building combined vocabulary...")
    
    # Prepare and enhance data
    enhanced_data_paths = []
    for path in file_paths:
        text = Path(path).read_text(encoding='utf-8')
        text = augment_data(text)  # Apply data augmentation
        text = apply_response_templates(text)  # Apply templates
        
        # Save enhanced data to temporary files in the working directory
        file_name = Path(path).name
        enhanced_path = f"/kaggle/working/{file_name}.enhanced"
        Path(enhanced_path).write_text(text)
        enhanced_data_paths.append(enhanced_path)

    
    # Organize by topic
    organized_data = organize_by_topic(enhanced_data_paths)
    organized_path = "/kaggle/working/organized_data.txt"
    Path(organized_path).write_text(organized_data)
    
    # Sort by complexity for curriculum learning
    sorted_data = sort_by_complexity(organized_data)
    sorted_path = "/kaggle/working/sorted_data.txt"
    Path(sorted_path).write_text(sorted_data)
    
    # Now build vocabulary from enhanced data
    chars, stoi, itos = build_combined_vocab([sorted_path])
    print(f"Vocabulary size: {len(chars)}")
    
    # Train the model with enhanced data
    print("Starting model training...")
    model, train_losses, val_losses = train_model([sorted_path], stoi, itos)
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('/kaggle/working/training_progress.png')
    
    # Evaluate with comprehensive metrics
    results, avg_relevance, avg_coherence = evaluate_model_quality(model, stoi, itos)
    
    # Standard evaluation
    evaluate_model(model, stoi, itos)
    
    # Start enhanced interactive mode
    interactive_coach_enhanced(model, stoi, itos)

if _name_ == "_main_":
    main()
