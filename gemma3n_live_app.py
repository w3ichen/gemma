#!/usr/bin/env python3
"""
Gemma 3n Live Video/Audio Processing App
A Gradio interface for real-time multimodal AI interaction using Gemma 3n
"""

import gradio as gr
import torch
import cv2
import numpy as np
import io
import tempfile
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login, whoami
import threading
import time
from datetime import datetime

class Gemma3nLiveApp:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.conversation_history = []
        self.authenticated = False
        
    def authenticate_huggingface(self, token):
        """Authenticate with Hugging Face using a token"""
        try:
            if not token or token.strip() == "":
                return "❌ Please provide a valid Hugging Face token"
            
            # Login with the token
            login(token=token.strip())
            
            # Verify authentication
            user_info = whoami()
            self.authenticated = True
            return f"✅ Successfully authenticated as: {user_info['name']}"
            
        except Exception as e:
            self.authenticated = False
            return f"❌ Authentication failed: {str(e)}"
    
    def check_model_access(self, model_size="E4B"):
        """Check if we can access the model"""
        try:
            model_id = f"google/gemma-3n-{model_size.lower()}-it"
            
            # Try to load just the config to test access
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            return f"✅ Access confirmed for {model_id}"
            
        except Exception as e:
            return f"❌ Cannot access {model_id}: {str(e)}"
        
    def load_model(self, model_size="E4B"):
        """Load the Gemma 3n model
        Blog: https://huggingface.co/blog/gemma3n
        Models: https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4
            - google/gemma-3n-E4B-it
            - google/gemma-3n-E2B-it
            - google/gemma-3n-E2B
            - google/gemma-3n-E4B
        """
        try:
            if not self.authenticated:
                return "❌ Please authenticate with Hugging Face first"
            
            model_id = f"google/gemma-3n-{model_size.lower()}-it"
            print(f"Loading {model_id}...")
            
            # Check available device and handle MPS memory limitations
            if torch.backends.mps.is_available():
                device = "cpu"  # Use CPU for loading, then move to MPS if possible
                dtype = torch.float16
                print("⚠️ Using CPU loading due to MPS memory limitations")
            elif torch.cuda.is_available():
                device = "cuda"
                dtype = torch.bfloat16
            else:
                device = "cpu"
                dtype = torch.float32
            
            print(f"Loading on: {device}, dtype: {dtype}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model with aggressive memory optimization
            try:
                # First attempt: Load on CPU with maximum memory savings
                print("🔄 Loading model on CPU with memory optimization...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    max_memory={0: "6GB"},  # Limit memory usage
                    offload_folder="./offload",  # Use disk offloading if needed
                ).eval()
                
                # Try to move to MPS in smaller chunks if available
                if torch.backends.mps.is_available():
                    try:
                        print("🔄 Attempting to move model to MPS...")
                        self.model = self.model.to("mps")
                        device = "mps"
                        print("✅ Successfully moved to MPS")
                    except Exception as mps_error:
                        print(f"⚠️ MPS failed, staying on CPU: {mps_error}")
                        device = "cpu"
                
            except Exception as load_error:
                # Fallback: Force E2B model if E4B fails
                if model_size == "E4B":
                    print("⚠️ E4B failed, trying E2B model...")
                    model_id = "google/gemma-3n-e2b-it"
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                    ).eval()
                    
                    if torch.backends.mps.is_available():
                        try:
                            self.model = self.model.to("mps")
                            device = "mps"
                        except:
                            device = "cpu"
                else:
                    raise load_error
            
            self.model_loaded = True
            
            # Clean up any temporary offload files
            if os.path.exists("./offload"):
                import shutil
                shutil.rmtree("./offload", ignore_errors=True)
            
            return f"✅ Model {model_id} loaded successfully on {device}!"
            
        except Exception as e:
            error_msg = str(e)
            if "Invalid buffer size" in error_msg:
                return f"❌ Memory error: Your system doesn't have enough RAM for this model. Try:\n1. Close other apps\n2. Use E2B model instead\n3. Restart your computer\n\nDetailed error: {error_msg}"
            else:
                return f"❌ Error loading model: {error_msg}"
    
    def process_image_frame(self, image, prompt="Describe what you see in this image."):
        """Process a single image frame"""
        if not self.model_loaded:
            return "⚠️ Model not loaded. Please authenticate and load the model first."
        
        if image is None:
            return "No image provided"
        
        try:
            # Prepare the message format for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process through the model
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Handle different devices
            if torch.backends.mps.is_available() and next(self.model.parameters()).device.type == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            else:
                inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                generation = generation[:, input_len:]
            
            decoded = self.processor.batch_decode(generation, skip_special_tokens=True)[0]
            
            # Add to conversation history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.conversation_history.append(f"[{timestamp}] Vision: {decoded}")
            
            return decoded
            
        except Exception as e:
            return f"❌ Error processing image: {str(e)}"
    
    def process_audio(self, audio_file, prompt="Transcribe this audio:"):
        """Process audio input"""
        if not self.model_loaded:
            return "⚠️ Model not loaded. Please authenticate and load the model first."
        
        if audio_file is None:
            return "No audio provided"
        
        try:
            # Prepare the message format for audio
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "audio": audio_file}
                    ]
                }
            ]
            
            # Process through the model
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Handle different devices
            if torch.backends.mps.is_available() and next(self.model.parameters()).device.type == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            else:
                inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                generation = generation[:, input_len:]
            
            decoded = self.processor.batch_decode(generation, skip_special_tokens=True)[0]
            
            # Add to conversation history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.conversation_history.append(f"[{timestamp}] Audio: {decoded}")
            
            return decoded
            
        except Exception as e:
            return f"❌ Error processing audio: {str(e)}"
    
    def process_multimodal(self, image, audio, custom_prompt):
        """Process both image and audio together"""
        if not self.model_loaded:
            return "⚠️ Model not loaded. Please authenticate and load the model first."
        
        try:
            content = []
            
            if custom_prompt:
                content.append({"type": "text", "text": custom_prompt})
            else:
                content.append({"type": "text", "text": "Describe what you see and hear:"})
            
            if image is not None:
                content.append({"type": "image", "image": image})
            
            if audio is not None:
                content.append({"type": "audio", "audio": audio})
            
            if len(content) == 1:  # Only prompt, no media
                return "Please provide either an image or audio input."
            
            messages = [{"role": "user", "content": content}]
            
            # Process through the model
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Handle different devices
            if torch.backends.mps.is_available() and next(self.model.parameters()).device.type == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            else:
                inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                generation = generation[:, input_len:]
            
            decoded = self.processor.batch_decode(generation, skip_special_tokens=True)[0]
            
            # Add to conversation history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.conversation_history.append(f"[{timestamp}] Multimodal: {decoded}")
            
            return decoded
            
        except Exception as e:
            return f"❌ Error processing multimodal input: {str(e)}"
    
    def get_conversation_history(self):
        """Get the conversation history as a string"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        return "\n".join(self.conversation_history[-10:])  # Show last 10 interactions
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."

# Initialize the app
app = Gemma3nLiveApp()

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Gemma 3n Live Video/Audio App", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎥 Gemma 3n Live Multimodal AI Assistant
        
        This app provides real-time interaction with Google's Gemma 3n model using your camera and microphone.
        
        **Features:**
        - 📸 Live image analysis from your camera
        - 🎤 Audio transcription and understanding
        - 🧠 Multimodal processing (image + audio together)
        - 💬 Conversation history tracking
        
        **⚠️ First Time Setup Required:**
        1. Get a Hugging Face token at: https://huggingface.co/settings/tokens
        2. Request access to Gemma models at: https://huggingface.co/google/gemma-3n-e4b-it
        3. Enter your token below and authenticate
        """)
        
        # Authentication section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔐 Step 1: Authenticate with Hugging Face")
                hf_token = gr.Textbox(
                    label="Hugging Face Token", 
                    placeholder="hf_...", 
                    type="password",
                    info="Get your token from: https://huggingface.co/settings/tokens"
                )
                auth_btn = gr.Button("🔑 Authenticate", variant="primary")
                auth_status = gr.Textbox(label="Authentication Status", interactive=False)
        
        auth_btn.click(app.authenticate_huggingface, inputs=[hf_token], outputs=[auth_status])
        
        # Model loading section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🚀 Step 2: Load Model")
                model_size = gr.Dropdown(
                    choices=["E2B", "E4B"], 
                    value="E2B", 
                    label="Model Size",
                    info="E2B: 2GB RAM, faster | E4B: 3GB RAM, better quality"
                )
                with gr.Row():
                    check_access_btn = gr.Button("🔍 Check Access")
                    load_btn = gr.Button("🚀 Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)
        
        check_access_btn.click(app.check_model_access, inputs=[model_size], outputs=[model_status])
        load_btn.click(app.load_model, inputs=[model_size], outputs=[model_status])
        
        gr.Markdown("---")
        
        # Main interface tabs
        with gr.Tabs():
            # Live Vision Tab
            with gr.TabItem("📸 Live Vision"):
                with gr.Row():
                    with gr.Column():
                        camera_input = gr.Image(
                            sources=["webcam"], 
                            label="Camera Feed",
                            streaming=True
                        )
                        vision_prompt = gr.Textbox(
                            value="Describe what you see in this image.",
                            label="Vision Prompt"
                        )
                        analyze_btn = gr.Button("🔍 Analyze Image")
                    
                    with gr.Column():
                        vision_output = gr.Textbox(
                            label="AI Response",
                            lines=8,
                            interactive=False
                        )
                
                analyze_btn.click(
                    app.process_image_frame,
                    inputs=[camera_input, vision_prompt],
                    outputs=[vision_output]
                )
            
            # Live Audio Tab
            with gr.TabItem("🎤 Live Audio"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            label="Microphone",
                            type="filepath"
                        )
                        audio_prompt = gr.Textbox(
                            value="Transcribe this audio:",
                            label="Audio Prompt"
                        )
                        transcribe_btn = gr.Button("🎧 Process Audio")
                    
                    with gr.Column():
                        audio_output = gr.Textbox(
                            label="AI Response",
                            lines=8,
                            interactive=False
                        )
                
                transcribe_btn.click(
                    app.process_audio,
                    inputs=[audio_input, audio_prompt],
                    outputs=[audio_output]
                )
            
            # Multimodal Tab
            with gr.TabItem("🎬 Multimodal (Vision + Audio)"):
                with gr.Row():
                    with gr.Column():
                        multi_image = gr.Image(sources=["webcam"], label="Camera")
                        multi_audio = gr.Audio(sources=["microphone"], label="Microphone", type="filepath")
                        multi_prompt = gr.Textbox(
                            value="Describe what you see and hear:",
                            label="Custom Prompt"
                        )
                        multi_btn = gr.Button("🚀 Process Both")
                    
                    with gr.Column():
                        multi_output = gr.Textbox(
                            label="AI Response",
                            lines=10,
                            interactive=False
                        )
                
                multi_btn.click(
                    app.process_multimodal,
                    inputs=[multi_image, multi_audio, multi_prompt],
                    outputs=[multi_output]
                )
        
        gr.Markdown("---")
        
        # Conversation History
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Conversation History")
                history_output = gr.Textbox(
                    label="Recent Interactions",
                    lines=8,
                    interactive=False
                )
                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 Refresh History")
                    clear_history_btn = gr.Button("🗑️ Clear History")
        
        refresh_history_btn.click(
            app.get_conversation_history,
            outputs=[history_output]
        )
        
        clear_history_btn.click(
            app.clear_history,
            outputs=[history_output]
        )
        
        # Instructions
        gr.Markdown("""
        ## 📋 How to Use:
        
        1. **Authenticate**: Get HF token and request model access, then authenticate
        2. **Load Model**: Choose E2B (faster) or E4B (better quality) and click "Load Model"
        3. **Live Vision**: Take photos with your camera and get AI descriptions
        4. **Live Audio**: Record audio clips and get transcriptions/analysis
        5. **Multimodal**: Combine both image and audio for richer interactions
        6. **History**: Track your conversation history with the AI
        
        **Tips:**
        - E2B model is recommended for most users (faster, less RAM)
        - E4B model provides better quality but needs more RAM
        - You can customize prompts to get different types of responses
        - The multimodal tab lets you combine vision and audio for natural conversations
        
        **Troubleshooting:**
        - If you get authentication errors, make sure you have access to the Gemma models
        - For Mac users: The app will automatically use MPS acceleration if available
        """)
    
    return demo

if __name__ == "__main__":
    # Check system info
    print("🖥️  System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("Open http://localhost:7860/ to use the app")
    
    # Create and launch the interface
    demo = create_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow access from other devices on network
        server_port=7860,
        share=False,  # Set to True if you want a public URL
        debug=True,
        show_error=True
    )