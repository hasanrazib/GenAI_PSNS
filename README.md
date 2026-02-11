# 📚 PSNS: Personal Study Notes Searcher



## 📖 Project Overview

**PSNS** is a Hybrid Retrieval-Augmented Generation (RAG) tool designed to help students efficiently search, summarize, and query scattered study materials. Unlike standard document chaters, PSNS specializes in:

* **Lecture Slides (PDF):** Extracts text and context from university slides.
* **Handwritten/Image Notes:** Uses **OCR (Tesseract)** to read text from images (PNG, JPG).
* **Hybrid AI Engine:** Switches between **OpenAI GPT-4o** (Cloud/Fast) and **Llama 3** (Local/Private).

---

## ✨ Key Features

* **⚡ Dual Mode:** Default runs on **GPT-4o** for speed. Can be toggled to **Local Llama 3** for privacy.
* **👁️ OCR Integration:** Built-in Tesseract OCR support for reading text from images/screenshots.
* **🔍 Source Verification:** Shows the exact page image or slide used to generate the answer.
* **🧠 Smart Chunking:** Optimized text chunking for faster processing on limited hardware.

---

## ⚙️ Configuration: Cloud vs. Local Mode

By default, the app is configured to use **OpenAI GPT-4o** because it is faster and more stable for demonstrations. However, you can switch to **Local Llama 3** easily.

To change the mode, open `app.py` and change **Line 24**:

```python
# 🔵 Set to False = Use OpenAI GPT-4o (Fast, Paid, Requires API Key) - DEFAULT
# 🟢 Set to True = Use Local Ollama (Free, Private, No API Key needed)

ENABLE_LOCAL_MODE = False

🛠️ Installation & Setup (Local - Recommended)

Running locally is recommended for the best performance, especially if your hardware resources are limited.
1. Prerequisites (Must Install)

Before running the app, ensure you have the following installed:

    Python 3.9+

    Tesseract OCR Engine: (Mandatory for image processing)

        Windows: Download Installer

        (Note: Install in default path: C:\Program Files\Tesseract-OCR\tesseract.exe)

        Mac: brew install tesseract

        Linux: sudo apt-get install tesseract-ocr

    Ollama (Only if using Local Mode):

        Download from ollama.com.

        Run this command in terminal to pull the model: ollama pull llama3

2. Install Dependencies

Open your terminal in the project folder:
Bash

# Create a virtual environment (Optional but recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

3. Configure API Key

Create a file named .env in the root folder and add your OpenAI API Key:
Code snippet

OPENAI_API_KEY=sk-proj-your-key-here...

(Note: If ENABLE_LOCAL_MODE = True, this key is not required).
4. Run the App 🚀

Execute the following command in your terminal:
Bash

streamlit run app.py

🐳 Run with Docker (Alternative)

You can run the app using Docker, but please note the performance warning below.

⚠️ Performance Warning: Running LLMs or Heavy OCR tasks inside Docker can be significantly slower than running locally, especially on Windows/Mac, due to virtualization overhead and lack of direct GPU access. For the smoothest experience, use the Local Setup above.

    Build and Run:
    Bash

    docker-compose up --build

    Access: Open your browser at http://localhost:8501.

📂 Project Structure
Plaintext

PSNS_Project/
├── app.py                 # Main Application Logic
├── requirements.txt       # Python Dependencies
├── Dockerfile             # Docker Configuration
├── docker-compose.yml     # Docker Compose
├── .env                   # API Keys (Not included in zip)
└── temp_files/            # Temporary storage for uploaded docs

❓ Troubleshooting

    Error: "Tesseract is not installed or it's not in your PATH"

        Make sure you installed Tesseract OCR using the Windows Installer.

        Check app.py line 30 to ensure the path matches your installation.

    Error: "ConnectionRefused" in Local Mode

        Make sure Ollama is running in the background.

        Ensure you have downloaded the model using ollama pull llama3.

    App is slow to respond?

        If using Local Mode (Llama 3), performance depends on your CPU/RAM.

        Switch to OpenAI Mode (ENABLE_LOCAL_MODE = False) for instant responses.