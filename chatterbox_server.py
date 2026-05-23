#!/usr/bin/env python3
"""
Minimal HTTP server that wraps Chatterbox TTS.
Run: python chatterbox_server.py
Then the web app can call POST http://localhost:5050/tts
"""

import io
import torch
import torchaudio
from flask import Flask, request, Response
from chatterbox.tts import ChatterboxTTS

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox on {device}...")
model = ChatterboxTTS.from_pretrained(device=device)
print("Model ready. Server starting on http://localhost:5050")


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


import subprocess
import json
import base64
import os
import tempfile

@app.route("/tts", methods=["GET", "POST", "OPTIONS"])
def tts():
    if request.method == "OPTIONS":
        return "", 200

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
    else:
        text = request.args.get("q", request.args.get("text", "")).strip()
        
    if not text:
        return {"error": "No text provided"}, 400

    exaggeration = float(request.args.get("exaggeration", 0.5) if request.method == "GET" else data.get("exaggeration", 0.5))
    cfg_weight   = float(request.args.get("cfg_weight", 0.5) if request.method == "GET" else data.get("cfg_weight", 0.5))

    print(f"Generating: {text[:60]}...")
    
    try:
        wav = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight)
        
        # Save to a temporary file for Rhubarb
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
            
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_file:
            json_path = json_file.name

        # Save audio to disk
        torchaudio.save(wav_path, wav, model.sr, format="WAV")
        
        # Run Rhubarb
        rhubarb_bin = "/Users/omidshojaeianzanjani/Documents/GreenTech/Jarvis/backend/bin/rhubarb"
        subprocess.run([rhubarb_bin, "-f", "json", "-o", json_path, wav_path, "-r", "phonetic"], check=True)
        
        # Read files
        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            
        with open(json_path, "r") as f:
            lipsync_data = json.load(f)
            
        # Cleanup
        os.remove(wav_path)
        os.remove(json_path)
        
        return {"audio": audio_b64, "lipsync": lipsync_data}
        
    except Exception as e:
        print(f"Error generating TTS: {e}")
        return {"error": str(e)}, 500


@app.route("/health")
def health():
    return {"status": "ok", "device": device}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
