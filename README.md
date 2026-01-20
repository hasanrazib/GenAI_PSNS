# 📚 PSNS: Personal Study Notes Searcher

**Course:** Generative AI (194.207) - TU Wien
**Group:** 181
**Members:** Al-Mamun Abdullah | Hasan Razib

---

## 📖 Project Overview
**PSNS** is a Retrieval-Augmented Generation (RAG) tool designed to help students efficiently search and summarize scattered study materials. Unlike standard PDF chaters, PSNS supports:
* **Lecture Slides (PDF)**
* **Handwritten Notes (Images/Screenshots)** via OCR integration.

---

## 🐳 Run with Docker (Recommended)

If you have Docker installed, you can run the app without installing dependencies manually.

1. **Build and Run:**
   ```bash
   docker-compose up --build


## 🛠️ Setup Instructions (Local)

To run this application on your local machine, please follow these steps:

### 1. Prerequisites
* **Python 3.9+**
* **Tesseract OCR Engine:** (Mandatory for image processing)
    * **Windows:** [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki)
    * **Mac:** `brew install tesseract`
    * **Linux:** `sudo apt-get install tesseract-ocr`

> **Note for Windows Users:**
> The app expects Tesseract at: `C:\Program Files\Tesseract-OCR\tesseract.exe`
> If you installed it elsewhere, please update the path in `app.py` (Line 18).

### 2. Install Dependencies
Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt

3. Configure API Key

    Create a new file named .env in this folder.

    Open the file and add your OpenAI API Key:
    Code snippet

    OPENAI_API_KEY=sk-proj-...

4. Run the App

Execute the following command:
Bash

streamlit run app.py

Note: For technical details, architecture, and evaluation results, please refer to the included Report.pdf.

