#!/usr/bin/env python3
"""
Gemma 3n Live Video/Audio Processing App - Mac Continuous Streaming
Uses Gradio's native streaming for Mac compatibility
Blog: https://huggingface.co/blog/gemma3n
Transformers docs: https://huggingface.co/docs/transformers/index
Gradio docs: https://www.gradio.app/docs/gradio/interface
"""

import gradio as gr
import torch
import numpy as np
import tempfile
import os
import json
import threading
import time
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login, whoami
import pyttsx3

class Gemma3nMacStreamingApp:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.conversation_history = []
        self.authenticated = False
        
        # Streaming state
        self.is_streaming = False
        self.latest_image = None
        self.latest_audio = None
        self.processing_lock = threading.Lock()
        self.last_process_time = 0
        self.process_interval = 3  # Process every 3 seconds
        
        # Text-to-speech
        self.tts_engine = None
        self.tts_enabled = True
        self.speaking = False
        
        # Persistence
        self.config_file = "myaeye_config.json"
        self.model_cache_dir = "./myaeye_models"
        
        # Initialize TTS
        self.init_tts()
        
        # Load saved config
        self.load_config()
        
    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a natural voice
            selected_voice = None
            for voice in voices:
                if any(name in voice.name.lower() for name in ['samantha', 'alex', 'female']):
                    selected_voice = voice.id
                    break
            
            if selected_voice:
                print(f"Using voice: {selected_voice}")
                self.tts_engine.setProperty('voice', selected_voice)
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.9)
            
            print("✅ Text-to-speech initialized")
            
        except Exception as e:
            print(f"⚠️ TTS initialization failed: {e}")
            self.tts_engine = None
    
    def speak_response(self, text):
        """Convert text to speech and play it"""
        if not self.tts_engine or not self.tts_enabled or self.speaking:
            print("⚠️ TTS not initialized or disabled")
            return
            
        try:
            self.speaking = True
            
            # Clean up text for better speech
            clean_text = text.replace('*', '').replace('**', '').replace('#', '').replace('[', '').replace(']', '')
            
            # Speak in a separate thread
            def speak():
                try:
                    self.tts_engine.say(clean_text)
                    self.tts_engine.runAndWait()
                finally:
                    self.speaking = False
            
            speech_thread = threading.Thread(target=speak)
            speech_thread.daemon = True
            speech_thread.start()
            
        except Exception as e:
            print(f"Speech error: {e}")
            self.speaking = False
    
    def toggle_tts(self):
        """Toggle text-to-speech on/off"""
        self.tts_enabled = not self.tts_enabled
        status = "🔊 Voice ON" if self.tts_enabled else "🔇 Voice OFF"
        return status
    
    def load_config(self):
        """Load saved configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                # Auto-authenticate if token exists
                if config.get('hf_token'):
                    try:
                        login(token=config['hf_token'])
                        self.authenticated = True
                        print("✅ Auto-authenticated with saved token")
                    except:
                        print("⚠️ Saved token invalid, need to re-authenticate")
                        
                return config
        except Exception as e:
            print(f"Config load error: {e}")
        return {}
    
    def save_config(self, hf_token=None, model_size=None):
        """Save configuration"""
        try:
            config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            
            if hf_token:
                config['hf_token'] = hf_token
            if model_size:
                config['model_size'] = model_size
                
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
                
        except Exception as e:
            print(f"Config save error: {e}")
    
    def authenticate_huggingface(self, token):
        """Authenticate with Hugging Face"""
        try:
            if not token or token.strip() == "":
                return "❌ Please provide a valid Hugging Face token"
            
            login(token=token.strip())
            user_info = whoami()
            self.authenticated = True
            
            # Save token
            self.save_config(hf_token=token.strip())
            
            return f"✅ Successfully authenticated as: {user_info['name']} (Saved for future sessions)"
            
        except Exception as e:
            self.authenticated = False
            return f"❌ Authentication failed: {str(e)}"
    
    def load_model(self, model_size="E2B"):
        """Load the Gemma 3n model - Mac optimized"""
        try:
            if not self.authenticated:
                return "❌ Please authenticate with Hugging Face first"
            
            model_id = f"google/gemma-3n-{model_size.lower()}-it"
            os.makedirs(self.model_cache_dir, exist_ok=True)
            
            print(f"Loading {model_id}...")
            
            # Force CPU for Mac stability
            device = "cpu"
            dtype = torch.float32
            print(f"Loading on: {device} (Mac optimized)")
            
            # Load processor
            print("🔄 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                cache_dir=self.model_cache_dir
            )
            
            # Load model
            print("🔄 Loading model...")
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    cache_dir=self.model_cache_dir,
                ).eval()
                
            except Exception as load_error:
                if model_size == "E4B":
                    print("⚠️ E4B failed, trying E2B...")
                    model_id = "google/gemma-3n-e2b-it"
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                        cache_dir=self.model_cache_dir,
                    ).eval()
                else:
                    raise load_error
            
            self.model_loaded = True
            self.save_config(model_size=model_size)
            
            return f"✅ Model {model_id} loaded successfully! (Mac optimized)"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def process_inputs(self, image, audio, prompt):
        """Process current image and audio inputs with proper format handling"""
        if not self.model_loaded:
            return "⚠️ Model not loaded"
        
        try:
            content = [{"type": "text", "text": prompt}]
            has_media = False
            
            # Handle image input - convert to PIL if needed
            if image is not None:
                try:
                    # If image is a numpy array (from Gradio webcam), convert to PIL
                    if isinstance(image, np.ndarray):
                        if len(image.shape) == 3:  # RGB image
                            image = Image.fromarray(image.astype('uint8'))
                        else:
                            image = Image.fromarray(image.astype('uint8'))

                    content.append({"type": "image", "image": image})
                    has_media = True
                    print("📸 Image processed")
                except Exception as img_error:
                    print(f"Image processing error: {img_error}")
            
            # Handle audio input - validate audio data
            if audio is not None:
                try:
                    # Check if audio is a tuple (sample_rate, data) from Gradio
                    if isinstance(audio, tuple) and len(audio) == 2:
                        sample_rate, audio_data = audio
                        
                        # Check if audio has actual content (not just noise)
                        if isinstance(audio_data, np.ndarray) and len(audio_data) > 0:
                            # Check audio amplitude to see if it's actual speech
                            audio_amplitude = np.abs(audio_data).mean()
                            if audio_amplitude > 0.01:  # Threshold for actual speech vs noise
                                # Save audio to temporary file
                                temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                                
                                # Ensure audio is in the right format
                                if len(audio_data.shape) > 1:
                                    audio_data = audio_data.mean(axis=1)  # Convert stereo to mono
                                
                                # Normalize audio
                                audio_data = audio_data / np.max(np.abs(audio_data))
                                
                                # Save using a simple wave writer
                                import wave
                                with wave.open(temp_audio_file.name, 'wb') as wav_file:
                                    wav_file.setnchannels(1)  # Mono
                                    wav_file.setsampwidth(2)  # 16-bit
                                    wav_file.setframerate(sample_rate)
                                    # Convert to 16-bit integers
                                    audio_16bit = (audio_data * 32767).astype(np.int16)
                                    wav_file.writeframes(audio_16bit.tobytes())
                                
                                content.append({"type": "audio", "audio": temp_audio_file.name})
                                has_media = True
                                print(f"🎤 Audio processed: {audio_amplitude:.4f} amplitude, {len(audio_data)} samples")
                            else:
                                print("🔇 Audio too quiet (likely noise), skipping")
                        else:
                            print("🔇 No audio data received")
                    elif isinstance(audio, str) and os.path.exists(audio):
                        # Audio file path
                        content.append({"type": "audio", "audio": audio})
                        has_media = True
                        print("🎤 Audio file processed")
                except Exception as audio_error:
                    print(f"Audio processing error: {audio_error}")
            
            if not has_media:
                return "No valid media detected (image may be loading or audio too quiet)"
            
            messages = [{"role": "user", "content": content}]
            
            print("MESSAGES: ", messages)
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            input_len = inputs["input_ids"].shape[-1]
            inputs = inputs.to("cpu")
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                generation = generation[:, input_len:]
                print("Generation: ", generation)
            
            decoded = self.processor.batch_decode(generation, skip_special_tokens=True)[0]
            
            # Clean up temporary audio file
            for item in content:
                if item.get("type") == "audio" and isinstance(item.get("audio"), str):
                    if item["audio"].startswith("/tmp") or "tmp" in item["audio"]:
                        try:
                            os.unlink(item["audio"])
                        except:
                            pass
            
            # Speak the response
            self.speak_response(decoded)
            
            # Add to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.conversation_history.append(f"[{timestamp}] {decoded}")
            
            return decoded
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Full processing error: {error_details}")
            return f"❌ Processing error: {str(e)}"
    
    def continuous_process(self, image, audio, prompt):
        """Continuous processing function that gets called by Gradio"""
        if not self.is_streaming or not self.model_loaded:
            return "Not streaming or model not loaded"
        
        # Rate limiting - only process every few seconds
        current_time = time.time()
        if current_time - self.last_process_time < self.process_interval:
            return "Waiting for next processing cycle..."
        
        # Process with lock to avoid overlapping
        with self.processing_lock:
            self.last_process_time = current_time
            result = self.process_inputs(image, audio, prompt)
            return result
    
    def start_streaming(self, prompt):
        """Start streaming mode"""
        if not self.model_loaded:
            return "⚠️ Model not loaded", "Start Streaming"
        
        self.is_streaming = True
        self.last_process_time = 0  # Reset timer
        
        return "🔴 LIVE: Streaming started! AI will respond every few seconds...", "Stop Streaming"
    
    def stop_streaming(self):
        """Stop streaming mode"""
        self.is_streaming = False
        return "⏹️ Streaming stopped", "Start Streaming"
    
    def get_conversation_history(self):
        """Get conversation history"""
        if not self.conversation_history:
            return "No conversation history yet."
        return "\n".join(self.conversation_history[-8:])
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "History cleared."

# Initialize the app
app = Gemma3nMacStreamingApp()

def create_interface():
    """Create the Gradio interface with continuous streaming"""
    
    auth_status = "✅ Auto-authenticated" if app.authenticated else "Not authenticated"
    
    with gr.Blocks(title="myaeye - Continuous AI Video Call", theme=gr.themes.Soft()) as demo:
        
        # State for streaming
        streaming_state = gr.State(False)
        
        gr.Markdown("""
        # 🎥 myaeye - Continuous AI Video Call
        ### Real-time conversation with AI that sees and hears you!
        
        **🔴 Continuous Mode:** Start streaming for ongoing AI conversation
        
        **Features:**
        - 🎥 **AI sees you** through continuous webcam feed
        - 🎤 **AI hears you** through continuous microphone
        - 🗣️ **AI speaks back** with natural voice responses
        - 💾 **Remembers everything** - saves settings automatically
        """)
        
        # Setup section
        with gr.Accordion("🔐 Setup (Skip if configured)", open=not app.authenticated):
            with gr.Row():
                with gr.Column():
                    hf_token = gr.Textbox(
                        label="Hugging Face Token", 
                        placeholder="hf_...", 
                        type="password"
                    )
                    auth_btn = gr.Button("🔑 Authenticate")
                    auth_status_display = gr.Textbox(
                        value=auth_status,
                        label="Auth Status", 
                        interactive=False
                    )
            
            with gr.Row():
                model_size = gr.Dropdown(
                    choices=["E2B", "E4B"], 
                    value="E2B", 
                    label="Model Size"
                )
                load_btn = gr.Button("🚀 Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)
        
        gr.Markdown("---")
        
        # Main streaming interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Your Video Feed")
                
                # Continuous webcam
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Live Camera"
                )
                
                # Continuous microphone with better settings
                microphone = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    label="Live Microphone",
                    type="numpy"  # Get raw numpy data for better processing
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 AI Control Panel")
                
                conversation_prompt = gr.Textbox(
                    value="Have a natural conversation with me. Respond briefly to what you see and hear:",
                    label="Conversation Style",
                    lines=3
                )
                
                with gr.Row():
                    stream_btn = gr.Button(
                        "🔴 Start Streaming", 
                        variant="primary", 
                        size="lg"
                    )
                    voice_btn = gr.Button("🔊 Voice ON", size="lg")
                
                stream_status = gr.Textbox(
                    value="Ready to start...",
                    label="Stream Status",
                    interactive=False
                )
                
                # Processing output
                ai_response = gr.Textbox(
                    label="Latest AI Response",
                    lines=6,
                    interactive=False,
                    placeholder="AI responses will appear here..."
                )
        
        # Conversation history
        with gr.Accordion("📝 Conversation History", open=False):
            history_display = gr.Textbox(
                label="Recent Conversation",
                lines=8,
                interactive=False
            )
            with gr.Row():
                refresh_history_btn = gr.Button("🔄 Refresh")
                clear_history_btn = gr.Button("🗑️ Clear")
        
        # Continuous processing function
        def process_stream(image, audio, prompt, is_streaming_state):
            if is_streaming_state:
                return app.continuous_process(image, audio, prompt)
            return "Not streaming"
        
        # Toggle streaming
        def toggle_streaming(current_streaming, prompt):
            if not current_streaming:
                status, btn_text = app.start_streaming(prompt)
                return True, status, btn_text
            else:
                status, btn_text = app.stop_streaming()
                return False, status, btn_text
        
        # Event handlers
        auth_btn.click(
            app.authenticate_huggingface,
            inputs=[hf_token],
            outputs=[auth_status_display]
        )
        
        load_btn.click(
            app.load_model,
            inputs=[model_size],
            outputs=[model_status]
        )
        
        stream_btn.click(
            toggle_streaming,
            inputs=[streaming_state, conversation_prompt],
            outputs=[streaming_state, stream_status, stream_btn]
        )
        
        voice_btn.click(
            app.toggle_tts,
            outputs=[voice_btn]
        )
        
        # Continuous processing - triggers when webcam or microphone updates
        webcam.stream(
            process_stream,
            inputs=[webcam, microphone, conversation_prompt, streaming_state],
            outputs=[ai_response],
            time_limit=120,  # Allow longer sessions
            stream_every=3.0  # Process every 3 seconds for better stability
        )
        
        refresh_history_btn.click(
            app.get_conversation_history,
            outputs=[history_display]
        )
        
        clear_history_btn.click(
            app.clear_history,
            outputs=[history_display]
        )
        
    return demo

if __name__ == "__main__":
    print("🖥️  System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if app.authenticated:
        print("✅ Auto-authenticated from saved credentials")
    
    if app.tts_engine:
        print("🔊 Text-to-speech ready")
    
    demo = create_interface()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )