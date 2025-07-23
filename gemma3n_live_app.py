#!/usr/bin/env python3
"""
Mac-Optimized Gemma 3n Multimodal AI with MPS Memory Management
Addresses MPS backend memory issues on Apple Silicon and Intel Macs
"""


import os
import sys

# CRITICAL FIX: Remove problematic environment variables and use the safest approach
# The "invalid low watermark ratio 1.4" error occurs when PyTorch calculates internal ratios incorrectly
# Solution: Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (unlimited) as recommended by PyTorch team

if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
    del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
if 'PYTORCH_MPS_LOW_WATERMARK_RATIO' in os.environ:
    del os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO']

# Set the working configuration - this is the ONLY reliable setting for many systems
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit as recommended
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable CPU fallback for unsupported ops

# Additional memory optimizations
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

print("🔧 MPS Configuration:")
print(f"   PYTORCH_MPS_HIGH_WATERMARK_RATIO: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'Not set')}")
print(f"   PYTORCH_ENABLE_MPS_FALLBACK: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'Not set')}")


import gradio as gr
import torch
import numpy as np
import tempfile
import json
import threading
import time
import wave
import gc
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login, whoami
import pyttsx3
from typing import Optional, Tuple, List, Dict, Any

class MacOptimizedGemma3n:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.conversation_history = []
        self.authenticated = False
        
        # Memory management
        self.device = self._get_optimal_device()
        self.dtype = self._get_optimal_dtype()
        self.max_memory_gb = self._get_available_memory()
        
        # Streaming control
        self.is_streaming = False
        self.processing_lock = threading.Lock()
        self.last_process_time = 0
        self.process_interval = 3.0  # Increased interval for stability
        
        # Text-to-speech
        self.tts_engine = None
        self.tts_enabled = True
        self.speaking = False
        
        # Configuration
        self.config_file = "gemma3n_config.json"
        self.model_cache_dir = "./gemma3n_models"
        
        # Initialize components
        self.init_tts()
        self.load_config()
        
        print(f"🖥️ Device: {self.device}")
        print(f"🧮 Data type: {self.dtype}")
        print(f"💾 Available memory: ~{self.max_memory_gb:.1f}GB")
        
    def _get_optimal_device(self) -> str:
        """Determine the best device for this Mac with buffer size awareness"""
        try:
            if torch.backends.mps.is_available():
                # Test MPS with a small tensor first
                try:
                    test_tensor = torch.ones(1000, 1000, device='mps')  # Small test
                    del test_tensor
                    print("🍎 MPS basic test passed")
                    
                    # Check available memory more conservatively
                    import psutil
                    total_ram = psutil.virtual_memory().total / (1024**3)
                    if total_ram < 16:  # Less than 16GB total RAM
                        print("⚠️ Limited RAM detected - recommending CPU for stability")
                        return "cpu"
                    
                    return "mps"
                except Exception as e:
                    print(f"⚠️ MPS test failed: {e}")
                    return "cpu"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except Exception as e:
            print(f"⚠️ Device detection error: {e}")
            return "cpu"
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal data type based on device"""
        if self.device == "mps":
            # MPS works better with float32 for stability
            return torch.float32
        elif self.device == "cuda":
            return torch.bfloat16
        else:
            return torch.float32
    
    def _get_available_memory(self) -> float:
        """Estimate available memory with buffer size considerations"""
        try:
            if self.device == "mps":
                import psutil
                total_ram = psutil.virtual_memory().total / (1024**3)
                # Be much more conservative due to buffer size limits
                safe_limit = min(total_ram * 0.3, 4.0)  # Max 4GB and only 30% of RAM
                print(f"💾 Total RAM: {total_ram:.1f}GB, Ultra-safe MPS limit: {safe_limit:.1f}GB")
                return safe_limit
            elif self.device == "cuda":
                return torch.cuda.get_device_properties(0).total_memory / (1024**3) * 0.8
            else:
                import psutil
                return psutil.virtual_memory().available / (1024**3) * 0.5
        except Exception as e:
            print(f"⚠️ Memory estimation error: {e}")
            return 2.0  # Very conservative fallback
    
    def clear_memory(self):
        """Aggressively clear GPU/MPS memory"""
        try:
            if self.device == "mps":
                # Clear MPS cache
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                # Force garbage collection
                gc.collect()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            else:
                gc.collect()
        except Exception as e:
            print(f"Memory clear warning: {e}")
    
    def init_tts(self):
        """Initialize text-to-speech with error handling"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a good voice
                for voice in voices:
                    if any(name in voice.name.lower() for name in ['samantha', 'alex', 'female', 'karen']):
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.setProperty('rate', 175)
            self.tts_engine.setProperty('volume', 0.9)
            print("✅ Text-to-speech initialized")
            
        except Exception as e:
            print(f"⚠️ TTS initialization failed: {e}")
            self.tts_engine = None
    
    def speak_response(self, text: str):
        """Convert text to speech in a separate thread"""
        if not self.tts_engine or not self.tts_enabled or self.speaking:
            return
            
        def speak():
            try:
                self.speaking = True
                # Clean text for better speech
                clean_text = text.replace('*', '').replace('**', '').replace('#', '')
                clean_text = clean_text.replace('[', '').replace(']', '')
                
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
            finally:
                self.speaking = False
        
        speech_thread = threading.Thread(target=speak, daemon=True)
        speech_thread.start()
    
    def load_config(self) -> Dict[str, Any]:
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
                    except Exception as e:
                        print(f"⚠️ Auto-authentication failed: {e}")
                        
                return config
        except Exception as e:
            print(f"Config load error: {e}")
        return {}
    
    def save_config(self, hf_token: Optional[str] = None, model_size: Optional[str] = None):
        """Save configuration to file"""
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
                json.dump(config, f, indent=2)
                
        except Exception as e:
            print(f"Config save error: {e}")
    
    def authenticate_huggingface(self, token: str) -> str:
        """Authenticate with Hugging Face"""
        try:
            if not token or token.strip() == "":
                return "❌ Please provide a valid Hugging Face token"
            
            login(token=token.strip())
            user_info = whoami()
            self.authenticated = True
            
            # Save token for future use
            self.save_config(hf_token=token.strip())
            
            return f"✅ Successfully authenticated as: {user_info['name']}"
            
        except Exception as e:
            self.authenticated = False
            return f"❌ Authentication failed: {str(e)}"
    
    def load_model(self, model_size: str = "E2B") -> str:
        """Load Gemma 3n model with aggressive size controls"""
        try:
            if not self.authenticated:
                return "❌ Please authenticate with Hugging Face first"
            
            # Clear memory before loading
            self.clear_memory()
            
            # FORCE E2B on MPS to avoid buffer size issues
            if self.device == "mps":
                model_size = "E2B"
                print("🔒 Forcing E2B model on MPS to avoid buffer size errors")
            
            model_id = f"google/gemma-3n-{model_size.lower()}-it"
            os.makedirs(self.model_cache_dir, exist_ok=True)
            
            print(f"🔄 Loading {model_id} with buffer size limits...")
            
            # Load processor first
            print("📚 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                cache_dir=self.model_cache_dir
            )
            
            # Ultra-conservative model loading for MPS
            print("🧠 Loading model with strict memory controls...")
            
            model_kwargs = {
                "torch_dtype": self.dtype,
                "low_cpu_mem_usage": True,
                "cache_dir": self.model_cache_dir,
                "device_map": None,  # Manual placement
            }
            
            # Load on CPU first, then carefully move to MPS
            try:
                print("   Step 1: Loading on CPU...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    **model_kwargs
                ).eval()
                
                if self.device == "mps":
                    print("   Step 2: Attempting MPS transfer in chunks...")
                    try:
                        # Try to move model to MPS with error catching
                        self.model = self.model.to(self.device)
                        print("   ✅ Successfully moved to MPS")
                    except Exception as mps_error:
                        print(f"   ⚠️ MPS transfer failed: {mps_error}")
                        print("   📱 Keeping on CPU for stability")
                        self.device = "cpu"
                
                self.model_loaded = True
                self.save_config(model_size=model_size)
                self.clear_memory()
                
                return f"✅ Model loaded on {self.device}!"
                
            except Exception as load_error:
                # Complete CPU fallback
                print(f"⚠️ Model loading failed: {load_error}")
                if "Invalid buffer size" in str(load_error):
                    print("🔄 Buffer size error - forcing CPU mode...")
                    self.device = "cpu"
                    self.dtype = torch.float32
                    
                    model_kwargs["torch_dtype"] = self.dtype
                    
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_id,
                        **model_kwargs
                    ).eval()
                    
                    self.model_loaded = True
                    return f"✅ Model loaded on CPU (buffer size fallback)"
                else:
                    raise load_error
                    
        except Exception as e:
            self.model_loaded = False
            error_msg = f"❌ Error loading model: {str(e)}"
            
            if "Invalid buffer size" in str(e):
                error_msg += "\n\n💡 Buffer size error detected:"
                error_msg += "\n   This happens when the model is too large for MPS."
                error_msg += "\n   The app will automatically use CPU mode."
                error_msg += "\n   CPU mode is slower but will work reliably."
            
            print(error_msg)
            return error_msg
            
    def preprocess_audio(self, audio_input) -> Optional[str]:
        """Convert audio input to proper format for the model"""
        if audio_input is None:
            return None
            
        try:
            if isinstance(audio_input, tuple) and len(audio_input) == 2:
                sample_rate, audio_data = audio_input
                
                # Check if audio has actual content
                if isinstance(audio_data, np.ndarray) and len(audio_data) > 0:
                    # Check amplitude to filter out noise
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)  # Convert to mono
                    
                    audio_amplitude = np.abs(audio_data).mean()
                    if audio_amplitude > 0.005:  # Threshold for actual speech
                        # Create temporary WAV file
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        
                        # Normalize and convert to 16-bit
                        if audio_data.max() > 0:
                            audio_data = audio_data / np.max(np.abs(audio_data))
                        audio_16bit = (audio_data * 32767).astype(np.int16)
                        
                        # Save as WAV
                        with wave.open(temp_file.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)  # Mono
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes(audio_16bit.tobytes())
                        
                        print(f"🎤 Audio processed: {audio_amplitude:.4f} amplitude")
                        return temp_file.name
                    else:
                        print("🔇 Audio too quiet, skipping")
                        return None
            
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                return audio_input
                
        except Exception as e:
            print(f"Audio preprocessing error: {e}")
        
        return None
    
    def process_multimodal_input(self, image, audio, prompt: str) -> str:
        """Process multimodal input with Mac-optimized memory management"""
        if not self.model_loaded:
            return "⚠️ Model not loaded. Please load the model first."
        
        print(f"🎯 Starting processing - Image: {image is not None}, Audio: {audio is not None}")
        
        try:
            # Clear memory before processing
            self.clear_memory()
            
            # Prepare message content
            content = [{"type": "text", "text": prompt}]
            has_media = False
            
            # Handle image input with memory optimization
            if image is not None:
                try:
                    # Convert and resize image if too large
                    if isinstance(image, np.ndarray):
                        # Resize if image is too large to save memory
                        if image.shape[0] > 768 or image.shape[1] > 768:
                            from PIL import Image as PILImage
                            pil_img = PILImage.fromarray(image.astype('uint8'), 'RGB')
                            pil_img = pil_img.resize((768, 768), PILImage.Resampling.LANCZOS)
                            image_pil = pil_img
                        else:
                            image_pil = Image.fromarray(image.astype('uint8'), 'RGB')
                    else:
                        image_pil = image
                    
                    content.append({"type": "image", "image": image_pil})
                    has_media = True
                    print("📸 Image processed")
                    
                except Exception as img_error:
                    print(f"Image processing error: {img_error}")
            
            # Handle audio input
            audio_path = self.preprocess_audio(audio)
            if audio_path:
                content.append({"type": "audio", "audio": audio_path})
                has_media = True
                print("🎤 Audio processed")
            
            if not has_media:
                return "No valid media detected. Please ensure your camera and microphone are working."
            
            # Create message structure
            messages = [{"role": "user", "content": content}]
            
            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response with memory-conscious settings
            with torch.inference_mode():
                # generation = self.model.generate(
                #     **inputs,
                #     max_new_tokens=100,  # Reduced for memory efficiency
                #     do_sample=True,
                #     temperature=0.7,
                #     top_p=0.9,
                #     pad_token_id=self.processor.tokenizer.eos_token_id,
                #     eos_token_id=self.processor.tokenizer.eos_token_id,
                #     use_cache=True,  # Enable KV cache
                # )
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Even smaller
                    do_sample=False,    # Disable sampling for speed
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=False,    # Disable cache to save memory
                )
                
                # Extract only the new tokens
                generation = generation[:, input_len:]
            
            # Decode the response
            decoded = self.processor.batch_decode(generation, skip_special_tokens=True)[0]
            
            # Clean up temporary audio file
            if audio_path and audio_path != audio:
                try:
                    os.unlink(audio_path)
                except:
                    pass
            
            # Clear memory after processing
            self.clear_memory()
            
            # Speak the response
            self.speak_response(decoded)
            
            # Add to conversation history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.conversation_history.append({
                "timestamp": timestamp,
                "response": decoded
            })
            
            # Keep only last 5 conversations to save memory
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
            
            return decoded
            
        except RuntimeError as e:
            if "MPS backend out of memory" in str(e):
                self.clear_memory()
                return "⚠️ MPS memory exhausted. Try:\n1. Restart the app\n2. Use smaller images\n3. Switch to CPU mode"
            else:
                import traceback
                error_details = traceback.format_exc()
                print(f"Processing error: {error_details}")
                return f"❌ Error: {str(e)}"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Processing error: {error_details}")
            return f"❌ Error processing input: {str(e)}"
    
    def toggle_tts(self) -> str:
        """Toggle text-to-speech on/off"""
        self.tts_enabled = not self.tts_enabled
        return "🔊 Voice ON" if self.tts_enabled else "🔇 Voice OFF"
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        history_text = []
        for item in self.conversation_history[-3:]:  # Show last 3 to save memory
            history_text.append(f"[{item['timestamp']}] {item['response']}")
        
        return "\n\n".join(history_text)
    
    def clear_history(self) -> str:
        """Clear conversation history and memory"""
        self.conversation_history = []
        self.clear_memory()
        return "Conversation history cleared and memory freed."
    
    def get_memory_status(self) -> str:
        """Get current memory status"""
        try:
            if self.device == "mps":
                # For MPS, we can't easily get allocated memory, so estimate
                return f"Device: {self.device} | Estimated usage: Moderate"
            elif self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**3)
                cached = torch.cuda.memory_reserved() / (1024**3)
                return f"Device: {self.device} | Allocated: {allocated:.2f}GB | Cached: {cached:.2f}GB"
            else:
                import psutil
                memory = psutil.virtual_memory()
                return f"Device: {self.device} | RAM: {memory.percent}% used"
        except:
            return f"Device: {self.device} | Status: Unknown"

def create_mac_optimized_interface():
    """Create Mac-optimized Gradio interface"""
    
    # Initialize the AI system
    ai_system = MacOptimizedGemma3n()
    
    # Check authentication status
    auth_status = "✅ Auto-authenticated" if ai_system.authenticated else "❌ Not authenticated"
    
    with gr.Blocks(
        title="Mac-Optimized Gemma 3n AI", 
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .status-text { font-weight: bold; }
        .warning-text { color: #ff6b6b; }
        .success-text { color: #51cf66; }
        """
    ) as demo:
        
        gr.Markdown(f"""
        # 🍎 Mac-Optimized Gemma 3n Multimodal AI
        ### Specially optimized for Apple Silicon and Intel Macs
        
        **Current Configuration:**
        - 🖥️ Device: {ai_system.device.upper()}
        - 🧮 Data Type: {str(ai_system.dtype).split('.')[-1]}
        - 💾 Available Memory: ~{ai_system.max_memory_gb:.1f}GB
        - 🛡️ MPS High Watermark: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'Not set')}
        
        **Features:**
        - 🎥 Real-time video processing (memory-optimized)
        - 🎤 Continuous audio input with noise filtering
        - 🗣️ Natural voice responses
        - 💬 Conversation memory with auto-cleanup
        - 🧹 Automatic memory management
        """)
        
        # Memory status display
        memory_status = gr.Textbox(
            value=ai_system.get_memory_status(),
            label="System Status",
            interactive=False,
            elem_classes=["status-text"]
        )
        
        # Setup section
        with gr.Accordion("🔧 Setup & Configuration", open=not ai_system.authenticated):
            gr.Markdown("### Authentication")
            with gr.Row():
                hf_token = gr.Textbox(
                    label="Hugging Face Token", 
                    placeholder="hf_...", 
                    type="password",
                    info="Get your token from https://huggingface.co/settings/tokens"
                )
                auth_btn = gr.Button("🔑 Authenticate", variant="primary")
            
            auth_status_display = gr.Textbox(
                value=auth_status,
                label="Authentication Status", 
                interactive=False,
                elem_classes=["status-text"]
            )
            
            gr.Markdown("### Model Configuration")
            with gr.Row():
                model_size = gr.Dropdown(
                    choices=["E2B", "E4B"], 
                    value="E2B", 
                    label="Model Size",
                    info="E2B: Recommended for Mac (2GB) | E4B: Better quality (4GB+)"
                )
                load_btn = gr.Button("🚀 Load Model", variant="primary")
            
            model_status = gr.Textbox(
                label="Model Status", 
                interactive=False,
                elem_classes=["status-text"]
            )
            
            # Memory management controls
            with gr.Row():
                clear_memory_btn = gr.Button("🧹 Clear Memory", size="sm")
                restart_btn = gr.Button("🔄 Restart Recommended", size="sm", variant="secondary")
        
        gr.Markdown("---")
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Live Input")
                
                # Video input with streaming
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Live Camera Feed",
                    height=300
                )
                
                # Audio input with streaming
                microphone = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Live Microphone",
                    waveform_options=gr.WaveformOptions(show_recording_waveform=True)
                )
                
                # Control buttons
                with gr.Row():
                    voice_toggle_btn = gr.Button("🔊 Voice ON", size="sm")
                    clear_btn = gr.Button("🗑️ Clear History", size="sm")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 AI Response")
                
                # Conversation prompt
                conversation_prompt = gr.Textbox(
                    value="Please respond naturally and briefly to what you see and hear.",
                    label="Conversation Style",
                    lines=2,
                    info="Keep prompts short for better memory usage"
                )
                
                # AI response display
                ai_response = gr.Textbox(
                    label="Latest AI Response",
                    lines=8,
                    interactive=False,
                    placeholder="AI responses will appear here when you interact through camera and microphone..."
                )
                
                # Processing status
                processing_status = gr.Textbox(
                    label="Processing Status",
                    value="Ready - waiting for input...",
                    interactive=False,
                    max_lines=1
                )
        
        # Conversation history section
        with gr.Accordion("💬 Conversation History", open=False):
            history_display = gr.Textbox(
                label="Recent Conversations",
                lines=6,
                interactive=False,
                placeholder="Conversation history will appear here..."
            )
            refresh_history_btn = gr.Button("🔄 Refresh History")
        
        def process_streams(image, audio, prompt):
            """Process streaming inputs with fixed logic"""
            try:
                if not ai_system.model_loaded:
                    return "Please load the model first.", "⚠️ Model not loaded"
                
                # Check if we have any input at all
                if image is None and audio is None:
                    return "Waiting for camera/microphone input...", "🔍 No input detected"
                
                # Rate limiting - but allow first processing
                current_time = time.time()
                if (current_time - ai_system.last_process_time < ai_system.process_interval and 
                    ai_system.last_process_time > 0):
                    return ai_response.value or "Processing...", "⏱️ Rate limited - waiting..."
                
                # Process with lock and memory management
                with ai_system.processing_lock:
                    print(f"🔄 Processing at {datetime.now().strftime('%H:%M:%S')}")
                    ai_system.last_process_time = current_time
                    
                    response = ai_system.process_multimodal_input(image, audio, prompt)
                    memory_info = ai_system.get_memory_status()
                    
                    print(f"✅ Response generated: {response[:50]}...")
                    return response, f"✅ {datetime.now().strftime('%H:%M:%S')} | {memory_info}"
                    
            except Exception as e:
                print(f"❌ Stream processing error: {e}")
                ai_system.clear_memory()
                return f"Error: {str(e)}", "❌ Error occurred - memory cleared"
            
        # Set up streaming with Mac-optimized parameters
        webcam.stream(
            fn=process_streams,
            inputs=[webcam, microphone, conversation_prompt],
            outputs=[ai_response, processing_status],
            time_limit=30,   # Shorter for testing
            stream_every=2,  # More frequent processing
            concurrency_limit=1
        )
        
        # Event handlers
        auth_btn.click(
            fn=ai_system.authenticate_huggingface,
            inputs=[hf_token],
            outputs=[auth_status_display]
        )
        
        load_btn.click(
            fn=ai_system.load_model,
            inputs=[model_size],
            outputs=[model_status]
        )
        
        voice_toggle_btn.click(
            fn=ai_system.toggle_tts,
            outputs=[voice_toggle_btn]
        )
        
        refresh_history_btn.click(
            fn=ai_system.get_conversation_history,
            outputs=[history_display]
        )
        
        clear_btn.click(
            fn=ai_system.clear_history,
            outputs=[history_display]
        )
        
        clear_memory_btn.click(
            fn=lambda: ai_system.clear_memory() or "Memory cleared",
            outputs=[processing_status]
        )
        
        # Update memory status periodically
        def update_memory_status():
            return ai_system.get_memory_status()
        
        # Set up periodic memory status updates
        demo.load(
            update_memory_status,
            outputs=[memory_status]
        )
    
    return demo

if __name__ == "__main__":
    print("🍎 Starting Mac-Optimized Gemma 3n Multimodal AI...")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🖥️ CUDA available: {torch.cuda.is_available()}")
    print(f"🍎 MPS available: {torch.backends.mps.is_available()}")
    print(f"💾 MPS High Watermark Ratio: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'Not set')}")
    
    try:
        # Create and launch the interface
        demo = create_mac_optimized_interface()
        
        # Launch with Mac-optimized settings
        demo.launch(
            server_name="localhost",
            server_port=7860,
            share=False,
            debug=False,  # Disable debug for better performance
            show_error=True,
            quiet=False,
            inbrowser=True  # Auto-open browser
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Application error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure you have sufficient memory available")
        print("2. Close other memory-intensive applications")
        print("3. Try restarting your Mac")
        print("4. Consider using the E2B model instead of E4B")
        sys.exit(1)