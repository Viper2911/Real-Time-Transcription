# 🎙️ Real-Time Audio Transcription Web App (Whisper + Flask)

A lightweight and modern web application that allows users to upload audio files and receive highly accurate transcriptions in real time. It uses OpenAI's Whisper for speech recognition and Flask as the web framework.

---

## 🚀 Features

- 🔍 Upload audio files in `.mp3`, `.wav`, or `.m4a` formats  
- 💬 Get real-time transcription powered by Whisper  
- 🌐 Web-based interface with a clean, user-friendly design  
- ⚡ Fast and responsive thanks to local execution  
- 📦 Simple to install and run locally  

---

## 📸 Screenshot

> Clean and simple UI for uploading and transcribing audio.

_Add your screenshot here (e.g., `/assets/screenshot.png`)_

---

## 🧱 Tech Stack

- 🧠 OpenAI Whisper  
- 🌐 Flask (Python web framework)  
- 🎧 FFmpeg for audio handling  
- 🧪 Jinja2 templates (for HTML rendering)  

---

## 📦 Requirements

- Python 3.8 or newer  
- pip (Python package installer)  
- FFmpeg installed and accessible in PATH  

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/whisper-transcription-tool.git
cd whisper-transcription-tool
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate it:

- **Windows**
```bash
venv\Scripts\activate
```

- **Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

Make sure FFmpeg is installed and added to your system PATH.

- **Windows:** Download from https://ffmpeg.org/download.html  
- **Linux:**
```bash
sudo apt install ffmpeg
```
- **macOS:**
```bash
brew install ffmpeg
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 📂 Project Structure

```
whisper-transcription-tool/
│── static/
│── templates/
│── app.py
│── requirements.txt
│── README.md
```

---

## ✨ Future Improvements

- 🎙️ Live microphone recording  
- 🌍 Multi-language support  
- ☁️ Cloud deployment (Docker, AWS, etc.)  
- 📊 Downloadable transcripts  

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.
