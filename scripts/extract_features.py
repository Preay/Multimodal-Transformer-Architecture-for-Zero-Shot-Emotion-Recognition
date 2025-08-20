"""
Unified feature extractor for audio + vision.

- Audio features: Wav2Vec2 (mean-pooled embeddings)
- Vision features: ViT CLS token embedding from middle-frame face
- Supports: RAVDESS, CREMA-D, SAVEE
- Saves .pt files with matching filenames for both modalities

Usage:
  python scripts/extract_features.py \
    --dataset_name ravdess \
    --dataset_path /datasets/ravdess \
    --video_dir /datasets/ravdess_videos \
    --audio_out features/ravdess/audio \
    --vision_out features/ravdess/vision \
    --extract_audio \
    --extract_vision
"""

import os
import cv2
import torch
import argparse
import traceback
import torchaudio
import numpy as np
import sys
import logging
from tqdm import tqdm
from torchvision import transforms
from transformers import Wav2Vec2Processor, Wav2Vec2Model, ViTModel, ViTImageProcessor, AutoProcessor, AutoModel

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import build_dataset_df

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_features.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =========================
# AUDIO EXTRACTION (Wav2Vec2)
# =========================
def extract_wav2vec2(audio_path, processor, model, device):
    """Extract Wav2Vec2 embeddings from audio file with robust error handling."""
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
            
        # Load audio with error handling
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            return None
            
        # Handle different audio formats
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
            
        # Resample if needed
        if sr != 16000:
            try:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            except Exception as e:
                logger.error(f"Failed to resample audio {audio_path}: {e}")
                return None
                
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Ensure proper shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Process with Wav2Vec2
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(inputs.input_values.to(device))
            embedding = outputs.last_hidden_state.mean(dim=1).cpu()  # [1,768]
            
        return embedding
        
    except Exception as e:
        logger.error(f"Error in extract_wav2vec2 for {audio_path}: {e}")
        logger.error(traceback.format_exc())
        return None


def _load_audio_backbone(name: str, device):
    """Return (processor, model) for selected backbone."""
    name = (name or "wav2vec2_base").lower()
    if name == "wav2vec2_base":
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    elif name == "distilhubert":
        processor = AutoProcessor.from_pretrained("facebook/distilhubert")
        model = AutoModel.from_pretrained("facebook/distilhubert").to(device)
    else:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()
    return processor, model


def process_audio_features(df, audio_out_dir, device, audio_backbone: str = "wav2vec2_base"):
    """Process audio features with comprehensive error handling and logging."""
    # Ensure output directory exists
    os.makedirs(audio_out_dir, exist_ok=True)
    logger.info(f"Audio output directory: {audio_out_dir}")
    
    # Load models with error handling
    try:
        processor, model = _load_audio_backbone(audio_backbone, device)
        logger.info(f"DONE Audio backbone loaded: {audio_backbone}")
    except Exception as e:
        logger.error(f"Failed to load Wav2Vec2 models: {e}")
        return 0, len(df)

    success, fail = 0, 0
    failed_files = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio features"):
        audio_path = row["path"]
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = os.path.join(audio_out_dir, base_name + ".pt")

        # Skip if already exists
        if os.path.exists(save_path):
            continue

        try:
            emb = extract_wav2vec2(audio_path, processor, model, device)
            if emb is not None:
                torch.save(emb, save_path)
                success += 1
            else:
                # Save zero tensor as fallback
                torch.save(torch.zeros(1, 768), save_path)
                failed_files.append(audio_path)
                fail += 1
        except Exception as e:
            logger.error(f"❌ Audio failed: {audio_path} → {e}")
            # Save zero tensor as fallback
            torch.save(torch.zeros(1, 768), save_path)
            failed_files.append(audio_path)
            fail += 1

    logger.info(f"\nDONE AUDIO DONE: {success} files | ❌ Failed: {fail}")
    if failed_files:
        logger.warning(f"Failed files: {failed_files[:5]}...")  # Show first 5
    return success, fail


# =========================
# VISION EXTRACTION (ViT)
# =========================
def extract_face_frame(video_path, face_detector):
    """Extract face frame from video with robust error handling."""
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.warning(f"Video has no frames: {video_path}")
            cap.release()
            return None

        # Get middle frame
        mid_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            logger.warning(f"Failed to read frame from video: {video_path}")
            return None

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            logger.warning(f"No face detected in video: {video_path}")
            return None

        # Extract largest face with margin
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        margin = int(min(w, h) * 0.1)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        face_img = frame[y:y+h, x:x+w]
        
        # Validate face image
        if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
            logger.warning(f"Face image too small in video: {video_path}")
            return None
            
        return face_img
        
    except Exception as e:
        logger.error(f"Error in extract_face_frame for {video_path}: {e}")
        return None


def extract_vit_embedding(face_img, processor, model, device):
    """Extract ViT embeddings from face image with robust error handling."""
    try:
        # Ensure the image is in uint8 format with proper range
        if face_img.dtype != np.uint8:
            face_img = np.clip(face_img, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Validate image dimensions
        if rgb_face.shape[0] < 10 or rgb_face.shape[1] < 10:
            logger.warning(f"Face image too small: {rgb_face.shape}")
            return torch.zeros(1, 768)
        
        # Convert to PIL Image
        from PIL import Image
        pil_image = Image.fromarray(rgb_face)
        
        # Apply transforms with error handling
        try:
            transform_resize = transforms.Resize((224, 224))
            transform_tensor = transforms.ToTensor()
            
            # Resize
            pil_resized = transform_resize(pil_image)
            
            # Convert to tensor
            img_tensor = transform_tensor(pil_resized)
            
            # Ensure tensor is in [0, 1] range
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return torch.zeros(1, 768)
        
        # Process with ViT
        try:
            inputs = processor(images=img_tensor, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                feat = outputs.last_hidden_state[:, 0, :]  # CLS token

            return feat.cpu()
            
        except Exception as e:
            logger.error(f"Error in ViT processing: {e}")
            return torch.zeros(1, 768)
        
    except Exception as e:
        logger.error(f"Error in extract_vit_embedding: {e}")
        return torch.zeros(1, 768)


def _load_vision_backbone(name: str, device):
    """Return (processor, model) for selected vision backbone name."""
    name = (name or "vit_base").lower()
    if name == "vit_base":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    elif name == "vit_tiny":
        # Use DeiT tiny as a proxy small ViT
        processor = ViTImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224")
        model = ViTModel.from_pretrained("facebook/deit-tiny-patch16-224").to(device)
    elif name == "mobilevit":
        from transformers import MobileViTImageProcessor, MobileViTModel
        processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-xx-small")
        model = MobileViTModel.from_pretrained("apple/mobilevit-xx-small").to(device)
    else:
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    model.eval()
    return processor, model


def process_vision_features(df, video_dir, vision_out_dir, device, vision_backbone: str = "vit_base"):
    """Process vision features with comprehensive error handling and logging."""
    # Ensure output directory exists
    os.makedirs(vision_out_dir, exist_ok=True)
    logger.info(f"Vision output directory: {vision_out_dir}")
    
    # Load models with error handling
    try:
        processor, model = _load_vision_backbone(vision_backbone, device)
        
        # Load face detector
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(face_cascade_path):
            logger.error(f"Face cascade file not found: {face_cascade_path}")
            return 0, len(df), 0
            
        face_detector = cv2.CascadeClassifier(face_cascade_path)
        if face_detector.empty():
            logger.error("Failed to load face detector")
            return 0, len(df), 0
            
        logger.info(f"DONE Vision backbone loaded: {vision_backbone} | Face detector ready")
    except Exception as e:
        logger.error(f"Failed to load vision models: {e}")
        return 0, len(df), 0

    success, fail, no_face = 0, 0, 0
    failed_files = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting vision features"):
        audio_path = row["path"]
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = os.path.join(vision_out_dir, base_name + ".pt")

        # Skip if already exists
        if os.path.exists(save_path):
            continue

        # Map audio filename to video filename
        video_path = None
        try:
            parts = base_name.split('-')
            if len(parts) >= 7:
                # Replace the first part (03 -> 02 for video)
                video_parts = ['02'] + parts[1:]  # Keep everything except first part
                video_filename = '-'.join(video_parts) + '.mp4'
                
                # Extract actor number from the last part
                actor_num = parts[-1]  # e.g., "01" from "03-01-08-02-02-02-01"
                
                # Look for video file in the corresponding actor directory
                video_actor_dir = os.path.join(video_dir, f"Video_Speech_Actor_{actor_num.zfill(2)}", f"Actor_{actor_num.zfill(2)}")
                video_path = os.path.join(video_actor_dir, video_filename)
                
                # Minimal logging only
            else:
                logger.warning(f"Invalid filename format: {base_name}")
        except Exception as e:
            logger.error(f"Error mapping filename {base_name}: {e}")

        # Handle missing video
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Missing video for {base_name}: {video_path}")
            torch.save(torch.zeros(1, 768), save_path)
            failed_files.append(f"{base_name} (missing video)")
            fail += 1
            continue

        # Extract face frame
        face_img = extract_face_frame(video_path, face_detector)
        if face_img is None:
            logger.warning(f"⚠️ No face detected in {video_path}, saving placeholder")
            torch.save(torch.zeros(1, 768), save_path)
            failed_files.append(f"{base_name} (no face)")
            no_face += 1
            continue

        # Extract ViT embedding
        try:
            feat = extract_vit_embedding(face_img, processor, model, device)
            torch.save(feat, save_path)
            success += 1
        except Exception as e:
            logger.error(f"❌ Vision failed: {video_path} → {e}")
            torch.save(torch.zeros(1, 768), save_path)
            failed_files.append(f"{base_name} (extraction failed)")
            fail += 1

    logger.info(f"\nDONE VISION DONE: {success} | ⚠️ No face: {no_face} | ❌ Failed: {fail}")
    if failed_files:
        logger.warning(f"Failed files: {failed_files[:5]}...")  # Show first 5
    return success, fail, no_face


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Unified audio + vision feature extractor")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["ravdess","crema","savee"])
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset audio files")
    parser.add_argument("--video_dir", type=str, help="Path to dataset videos (for vision)")
    parser.add_argument("--audio_out", type=str, required=True, help="Output dir for audio features")
    parser.add_argument("--vision_out", type=str, help="Output dir for vision features")
    parser.add_argument("--extract_audio", action="store_true", help="Extract audio features")
    parser.add_argument("--extract_vision", action="store_true", help="Extract vision features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audio_backbone", type=str, default="wav2vec2_base", choices=["wav2vec2_base", "distilhubert"], help="Audio backbone")
    parser.add_argument("--vision_backbone", type=str, default="vit_base", choices=["vit_base", "vit_tiny", "mobilevit"], help="Vision backbone")
    args = parser.parse_args()

    logger.info("=== FEATURE EXTRACTION START ===")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Extract audio: {args.extract_audio}")
    logger.info(f"Extract vision: {args.extract_vision}")

    # Validate inputs
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        return
        
    if args.extract_vision and not args.video_dir:
        logger.error("Vision extraction requires --video_dir!")
        return
        
    if args.extract_vision and not args.vision_out:
        logger.error("Vision extraction requires --vision_out!")
        return

    # Load dataset df
    try:
        df = build_dataset_df(args.dataset_name, args.dataset_path)
        logger.info(f"DONE Loaded {len(df)} entries for {args.dataset_name.upper()}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Extract audio if requested
    if args.extract_audio:
        logger.info("\n=== AUDIO EXTRACTION START ===")
        try:
            process_audio_features(df, args.audio_out, device, audio_backbone=args.audio_backbone)
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            logger.error(traceback.format_exc())

    # Extract vision if requested
    if args.extract_vision:
        logger.info("\n=== VISION EXTRACTION START ===")
        try:
            process_vision_features(df, args.video_dir, args.vision_out, device, vision_backbone=args.vision_backbone)
        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            logger.error(traceback.format_exc())

    logger.info("=== FEATURE EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()
