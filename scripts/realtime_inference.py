"""
Real-time emotion prediction using microphone (audio) or webcam (video).
"""
import argparse
import torch
import numpy as np
import sounddevice as sd
import torchaudio
import cv2
from transformers import Wav2Vec2Processor, Wav2Vec2Model, ViTFeatureExtractor, ViTModel
from models.audio_zser_model import AudioZSERModel
from models.multimodal_zser_model import MultimodalZSERModel

AUDIO_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def predict_audio(model, processor, encoder, device):
    print("Speak now...")
    duration = 3  # seconds
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    waveform = torch.from_numpy(audio.squeeze()).float()
    input_values = processor(waveform.numpy(), sampling_rate=fs, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        outputs = encoder(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        logits = model(embedding.unsqueeze(0))
        pred = logits.argmax(dim=1).item()
    print(f"Predicted Emotion: {AUDIO_EMOTIONS[pred]}")


def predict_vision(model, extractor, encoder, device):
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (224, 224))
            inputs = extractor(images=face_img, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            with torch.no_grad():
                outputs = encoder(pixel_values)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
                logits = model(embedding.unsqueeze(0), embedding.unsqueeze(0))
                pred = logits.argmax(dim=1).item()
            cv2.putText(frame, f"Emotion: {AUDIO_EMOTIONS[pred]}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow('Webcam - Real-Time Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time emotion prediction using mic or webcam.")
    parser.add_argument('--mode', type=str, choices=['audio', 'vision'], required=True, help='Mode: audio or vision')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.mode == 'audio':
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device)
        model = AudioZSERModel(num_classes=8).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        while True:
            predict_audio(model, processor, encoder, device)
            if input("Press Enter to continue or 'q' to quit: ").strip().lower() == 'q':
                break
    else:
        extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
        model = MultimodalZSERModel(num_classes=8).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        predict_vision(model, extractor, encoder, device)

if __name__ == "__main__":
    main() 