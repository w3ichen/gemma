# Myaeye
## 🎥 Gemma 3n Live AI Assistant

A user-friendly desktop application that lets you have natural conversations with Google's advanced Gemma 3n AI using your camera and microphone. 

## 📥 Installation Guide

### Step 1: Install Conda (if you don't have it)

Conda is a tool that helps manage different software environments. Think of it like creating a separate, clean space for this AI app to live.

1. **Download Miniconda** (lighter version of Conda):
   - Go to: https://docs.conda.io/en/latest/miniconda.html
   - Download the version for your Mac (Intel or Apple Silicon)
   - Run the installer and follow the prompts

2. **Restart your Terminal** after installation

### Step 2: Set Up the AI Environment

Open Terminal (press `Cmd + Space`, type "Terminal", press Enter) and run these commands one by one:

```bash
# Create a special environment for this AI app
conda create -n ai python=3.11 -y

# Activate the environment (you'll need to do this every time)
conda activate ai

# Install the AI libraries (this may take a few minutes)
conda install pytorch torchvision torchaudio -c pytorch -y

# Install additional required packages
conda install -c conda-forge "numpy<2" pillow opencv -y

# Install the web interface and AI model tools
pip install gradio transformers timm accelerate safetensors huggingface_hub sounddevice soundfile pyttsx3 pygame librosa pyttsx3 mlx-vlm
```

**⏱️ This process takes 5-15 minutes depending on your internet speed.**

### Step 3: Download the App Code

1. **Download the app file**:
   - Save the Python code as `gemma3n_live_app.py` in your Downloads folder

### Step 4: Get Hugging Face Access

The AI models are hosted on Hugging Face (a platform for AI models). You need free access:

1. **Create a Hugging Face account**:
   - Go to: https://huggingface.co/join
   - Sign up with your email (it's completely free)

2. **Get an access token**:
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it "Gemma AI Access"
   - Select "Read" permissions
   - Click "Generate token"
   - **Copy and save this token** (starts with `hf_...`) - you'll need it later

3. **Request model access** (usually approved within minutes):
   - Go to: https://huggingface.co/google/gemma-3n-e2b-it
   - Click "Request access to this model"
   - Fill out the simple form explaining you want to use it for personal AI assistant
   - Do the same for: https://huggingface.co/google/gemma-3n-e4b-it

## 🚀 Running the App

### Every Time You Want to Use the App:

1. **Open Terminal**

2. **Navigate to your app folder**:
   ```bash
   cd Downloads  # or wherever you saved the file
   ```

3. **Activate the AI environment**:
   ```bash
   conda activate ai
   ```

4. **Start the app**:
   ```bash
   python gemma3n_live_app.py
   ```

5. **Open your web browser** and go to: http://localhost:7860

## 🔧 First-Time Setup in the App

### Step 1: Authenticate
1. **Enter your Hugging Face token** in the "Hugging Face Token" box
2. **Click "🔑 Authenticate"**
3. Wait for the green "✅ Successfully authenticated" message

### Step 2: Choose and Load AI Model
1. **Select model size**:
   - **E2B**: Faster, uses 2GB RAM, good quality (recommended for most users)
   - **E4B**: Slower, uses 3GB RAM, better quality

2. **Click "🔍 Check Access"** to verify you can download the model

3. **Click "🚀 Load Model"**

⚠️ **IMPORTANT**: The model download is **5-8GB** and can take **10-30 minutes** depending on your internet speed. The app will show "Loading..." - be patient and don't close it!

### Step 3: Start Using the AI!

Once loaded, you'll see "✅ Model loaded successfully!" Now you can:

- **📸 Live Vision Tab**: Take photos and ask questions
- **🎤 Live Audio Tab**: Record speech for transcription
- **🎬 Multimodal Tab**: Use camera AND microphone together for natural conversations
