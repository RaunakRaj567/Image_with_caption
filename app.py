import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Image That Actually Speaks",
    page_icon="🎙️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4A90E2, #9013FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .caption-box {
        background-color: #f8f9fa;
        border-left: 5px solid #4A90E2;
        padding: 1.2rem;
        border-radius: 6px;
        font-size: 1.35rem;
        font-weight: 600;
        color: #212529;
        margin-top: 1rem;
    }
    .metric-badge {
        display: inline-block;
        background: #e9ecef;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        color: #495057;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎙️ Image That Actually Speaks</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Neural Image Captioning with Pretrained ResNet50 & LSTM Decoder</div>', unsafe_allow_html=True)

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        return features.view(features.size(0), -1)

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, pad_idx=0):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.feature_projection = nn.Linear(2048, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        image_embeddings = self.feature_projection(features).unsqueeze(1)
        inputs = torch.cat((image_embeddings, embeddings), dim=1)
        outputs, _ = self.lstm(inputs)
        outputs = outputs[:, 1:, :]
        return self.fc(outputs)

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

# ==========================================
# RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_all_resources():
    # 1. Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. Vocab
    possible_vocab_paths = [
        "vocab.json",
        os.path.join(r"E:\Images", "vocab.json"),
        os.path.join(os.path.dirname(__file__), "vocab.json")
    ]
    vocab_path = next((p for p in possible_vocab_paths if os.path.exists(p)), None)
    if not vocab_path:
        st.error("Vocabulary file (vocab.json) not found! Please ensure vocab.json is saved.")
        st.stop()
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    word_to_idx = vocab_data["word_to_idx"]
    idx_to_word = {int(idx): word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    
    # 3. Model
    encoder = EncoderCNN()
    decoder = DecoderLSTM(embed_size=256, hidden_size=512, vocab_size=vocab_size, pad_idx=word_to_idx["<pad>"])
    model = ImageCaptioningModel(encoder, decoder).to(device)
    
    possible_ckpts = [
        os.path.join("model_checkpoints", "best_image_caption_model.pth"),
        os.path.join(r"E:\Images", "best_image_caption_model.pth"),
        "best_image_caption_model.pth"
    ]
    ckpt_path = next((p for p in possible_ckpts if os.path.exists(p)), None)
    if not ckpt_path:
        st.error("Model checkpoint (best_image_caption_model.pth) not found!")
        st.stop()
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 4. Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return model, word_to_idx, idx_to_word, device, transform, ckpt_path

model, word_to_idx, idx_to_word, device, image_transform, ckpt_path = load_all_resources()

# ==========================================
# INFERENCE ALGORITHMS
# ==========================================
def generate_greedy(img_tensor, model, word_to_idx, idx_to_word, device, max_length=37):
    """
    Generates caption using teacher-forcing aligned greedy decoding with start token.
    """
    model.eval()
    generated_words = []
    
    with torch.no_grad():
        features = model.encoder(img_tensor)
        # Step 0: Image projection initializes the LSTM state
        img_emb = model.decoder.feature_projection(features).unsqueeze(1)
        _, states = model.decoder.lstm(img_emb)
        
        # Step 1: Feed <start> token
        curr_token = torch.tensor([[word_to_idx["<start>"]]], dtype=torch.long, device=device)
        
        for _ in range(max_length):
            token_emb = model.decoder.embedding(curr_token)
            output, states = model.decoder.lstm(token_emb, states)
            logits = model.decoder.fc(output.squeeze(1))
            predicted_id = logits.argmax(dim=1).item()
            
            if predicted_id == word_to_idx["<end>"]:
                break
                
            word = idx_to_word[predicted_id]
            if word not in ["<start>", "<pad>", "<unk>"]:
                generated_words.append(word)
                
            curr_token = torch.tensor([[predicted_id]], dtype=torch.long, device=device)
            
    return " ".join(generated_words).capitalize()

def generate_beam_search(img_tensor, model, word_to_idx, idx_to_word, device, beam_width=5, max_length=37, repetition_penalty=1.2):
    """
    Generates high-quality caption using Beam Search with length penalty and repetition reduction.
    """
    model.eval()
    with torch.no_grad():
        features = model.encoder(img_tensor)
        img_emb = model.decoder.feature_projection(features).unsqueeze(1)
        _, states = model.decoder.lstm(img_emb)
        
        start_token = word_to_idx["<start>"]
        end_token = word_to_idx["<end>"]
        
        # Sequence tuple: (cumulative_log_prob, [token_ids], (h, c))
        sequences = [(0.0, [start_token], states)]
        
        for step in range(max_length):
            all_candidates = []
            all_completed = True
            
            for score, seq, st in sequences:
                if seq[-1] == end_token:
                    all_candidates.append((score, seq, st))
                    continue
                
                all_completed = False
                last_token = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                emb = model.decoder.embedding(last_token)
                output, new_st = model.decoder.lstm(emb, st)
                logits = model.decoder.fc(output.squeeze(1))
                log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
                
                # Apply repetition penalty for previously chosen tokens
                for past_tok in set(seq[1:]):
                    log_probs[past_tok] -= repetition_penalty
                
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)
                for k in range(beam_width):
                    cand_score = score + topk_probs[k].item()
                    cand_seq = seq + [topk_indices[k].item()]
                    all_candidates.append((cand_score, cand_seq, new_st))
            
            if all_completed:
                break
                
            # Rank with length normalization
            all_candidates.sort(key=lambda x: x[0] / (len(x[1])**0.7), reverse=True)
            sequences = all_candidates[:beam_width]
            
        best_seq = sequences[0][1]
        words = [idx_to_word[tok] for tok in best_seq if tok not in [word_to_idx["<start>"], word_to_idx["<end>"], word_to_idx["<pad>"], word_to_idx["<unk>"]]]
        return " ".join(words).capitalize()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Model Settings")
    st.markdown(f"**Hardware Device:** `{device}`")
    st.markdown(f"**Vocabulary Size:** `{len(word_to_idx)} tokens`")
    st.markdown(f"**Loaded Weights:** `{os.path.basename(ckpt_path)}`")
    
    st.divider()
    decoding_method = st.radio(
        "Decoding Strategy:",
        ["Beam Search (Recommended)", "Greedy Search (Fast)"],
        index=0
    )
    
    if "Beam Search" in decoding_method:
        beam_width = st.slider("Beam Width (k)", min_value=2, max_value=7, value=5)
        rep_penalty = st.slider("Repetition Penalty", min_value=0.5, max_value=2.0, value=1.2, step=0.1)
    else:
        beam_width = 1
        rep_penalty = 1.0

# ==========================================
# MAIN INTERACTION UI
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPEG / PNG)...",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("2. Generated Caption")
    if uploaded_file is not None:
        if st.button("✨ Generate Caption", type="primary", use_container_width=True):
            with st.spinner("Analyzing image features and decoding words..."):
                tensor_input = image_transform(pil_image).unsqueeze(0).to(device)
                
                if "Beam Search" in decoding_method:
                    caption = generate_beam_search(
                        tensor_input,
                        model,
                        word_to_idx,
                        idx_to_word,
                        device,
                        beam_width=beam_width,
                        repetition_penalty=rep_penalty
                    )
                else:
                    caption = generate_greedy(
                        tensor_input,
                        model,
                        word_to_idx,
                        idx_to_word,
                        device
                    )
                
                st.markdown(f'<div class="caption-box">"{caption}"</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<span class="metric-badge">Decoder: LSTM</span>'
                    f'<span class="metric-badge">Encoder: ResNet50</span>'
                    f'<span class="metric-badge">Strategy: {decoding_method.split()[0]}</span>',
                    unsafe_allow_html=True
                )
    else:
        st.info("👈 Please upload an image from the left panel to test the caption generator.")

st.divider()
st.markdown(
    "<center><small>Project: <b>Image That Actually Speaks</b> | PyTorch & Streamlit</small></center>",
    unsafe_allow_html=True
)
