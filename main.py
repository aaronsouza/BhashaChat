import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech
import base64
from dotenv import load_dotenv
from gtts import gTTS
import tempfile

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# List available models and select one
print("\n=== Available Gemini Models ===")
available_models = []
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            model_name = m.name.replace('models/', '')
            available_models.append(model_name)
            print(f"  ✓ {model_name}")
except Exception as e:
    print(f"Error listing models: {e}")

# Try to initialize model with fallback options
model = None
model_options = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-pro',
    'gemini-1.0-pro'
]

if available_models:
    model_options = available_models[:3] + model_options

for model_name in model_options:
    try:
        model = genai.GenerativeModel(model_name)
        test = model.generate_content("Hi")
        print(f"\n✅ Successfully using model: {model_name}\n")
        break
    except Exception as e:
        print(f"  ✗ Failed to use {model_name}: {str(e)[:100]}")
        continue

if model is None:
    raise ValueError("Could not initialize any Gemini model. Please check your API key.")

# Set Google Cloud credentials
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not GOOGLE_APPLICATION_CREDENTIALS:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found in environment variables")

if not os.path.isabs(GOOGLE_APPLICATION_CREDENTIALS):
    GOOGLE_APPLICATION_CREDENTIALS = os.path.join(os.path.dirname(__file__), GOOGLE_APPLICATION_CREDENTIALS)

if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    raise FileNotFoundError(f"Google Cloud credentials file not found: {GOOGLE_APPLICATION_CREDENTIALS}")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

# Initialize Google Speech client
speech_client = speech.SpeechClient()

def text_to_speech(text, language):
    """Convert text to speech and return base64 encoded audio"""
    try:
        tts_lang_map = {
            'english': 'en',
            'hindi': 'hi',
            'kannada': 'kn',
            'tamil': 'ta',
            'telugu': 'te',
            'malayalam': 'ml',
            'bengali': 'bn'
        }
        
        lang_code = tts_lang_map.get(language, 'en')
        
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            tts.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        with open(temp_audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        os.unlink(temp_audio_path)
        
        return audio_base64
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# Store conversation history
conversations = {}

class ConversationManager:
    def __init__(self, topic, lesson_content, language='english'):
        self.topic = topic
        self.lesson_content = lesson_content
        self.language = language.lower()
        self.history = []
        self.turn_count = 0
        self.max_turns = 10
        
    def get_system_prompt(self):
        language_configs = {
            'english': {
                'name': 'English',
                'instruction': 'Respond in English',
                'aspects': 'pronunciation, grammar, and vocabulary'
            },
            'hindi': {
                'name': 'Hindi (हिंदी)',
                'instruction': 'Respond in Hindi (Devanagari script). Use simple, conversational Hindi that a learner would understand.',
                'aspects': 'pronunciation (उच्चारण), grammar (व्याकरण), and vocabulary (शब्दावली)'
            },
            'kannada': {
                'name': 'Kannada (ಕನ್ನಡ)',
                'instruction': 'Respond in Kannada (Kannada script). Use simple, conversational Kannada that a learner would understand.',
                'aspects': 'pronunciation (ಉಚ್ಚಾರಣೆ), grammar (ವ್ಯಾಕರಣ), and vocabulary (ಶಬ್ದಕೋಶ)'
            },
            'tamil': {
                'name': 'Tamil (தமிழ்)',
                'instruction': 'Respond in Tamil (Tamil script). Use simple, conversational Tamil that a learner would understand.',
                'aspects': 'pronunciation (உச்சரிப்பு), grammar (இலக்கணம்), and vocabulary (சொல்வளம்)'
            },
            'telugu': {
                'name': 'Telugu (తెలుగు)',
                'instruction': 'Respond in Telugu (Telugu script). Use simple, conversational Telugu that a learner would understand.',
                'aspects': 'pronunciation (ఉచ్చారణ), grammar (వ్యాకరణం), and vocabulary (పదకోశం)'
            },
            'malayalam': {
                'name': 'Malayalam (മലയാളം)',
                'instruction': 'Respond in Malayalam (Malayalam script). Use simple, conversational Malayalam that a learner would understand.',
                'aspects': 'pronunciation (ഉച്ചാരണം), grammar (വ്യാകരണം), and vocabulary (പദസമ്പത്ത്)'
            },
            'bengali': {
                'name': 'Bengali (বাংলা)',
                'instruction': 'Respond in Bengali (Bengali script). Use simple, conversational Bengali that a learner would understand.',
                'aspects': 'pronunciation (উচ্চারণ), grammar (ব্যাকরণ), and vocabulary (শব্দভাণ্ডার)'
            }
        }
        
        config = language_configs.get(self.language, language_configs['english'])
        
        return f"""You are a {config['name']} language learning companion helping a student practice what they've learned.

Topic: {self.topic}
Lesson Content: {self.lesson_content}
Language: {config['name']}

IMPORTANT: {config['instruction']}

Your role:
1. Have a natural conversation with the student about the topic IN {config['name'].upper()}
2. Ask questions to assess their understanding
3. Correct mistakes gently and provide better alternatives
4. Track {config['aspects']} usage
5. After {self.max_turns} exchanges, provide a final score and detailed feedback

Conversation guidelines:
- Keep responses conversational and encouraging
- Ask follow-up questions to assess understanding
- Note any grammatical errors or pronunciation issues
- Be supportive and constructive
- Use simple, clear language appropriate for a learner

Current turn: {self.turn_count}/{self.max_turns}

If this is the final turn, provide a JSON response with:
{{
    "final_assessment": true,
    "score": <number out of 100>,
    "stars": <number 1-5>,
    "message": "<encouraging message IN {config['name'].upper()}>",
    "what_you_did_well": "<specific praise IN {config['name'].upper()}>",
    "improvement_tip": {{
        "what_they_said": "<exact problematic phrase>",
        "better_way": "<corrected phrase IN {config['name'].upper()}>",
        "explanation": "<why this is better IN {config['name'].upper()}>"
    }},
    "detailed_feedback": "<comprehensive feedback IN {config['name'].upper()}>"
}}

Otherwise, respond naturally IN {config['name'].upper()} to continue the conversation."""

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        if role == "user":
            self.turn_count += 1

    def get_response(self, user_message):
        self.add_message("user", user_message)
        
        context = self.get_system_prompt() + "\n\nConversation history:\n"
        for msg in self.history:
            context += f"{msg['role']}: {msg['content']}\n"
        
        response = model.generate_content(context)
        assistant_message = response.text.strip()
        
        self.add_message("assistant", assistant_message)
        
        is_final = self.turn_count >= self.max_turns
        
        if is_final:
            assessment_prompt = f"""Based on this conversation, provide a final assessment as ONLY valid JSON (no markdown, no backticks, no preamble):

Conversation:
{json.dumps(self.history[-20:], indent=2)}

Return ONLY this JSON structure, nothing else:
{{
    "score": 85,
    "stars": 4,
    "message": "Great job!",
    "what_you_did_well": "Your pronunciation was clear and you used polite language.",
    "improvement_tip": {{
        "what_they_said": "I want hot",
        "better_way": "I would like it hot, please",
        "explanation": "Adding 'I would like' and 'please' sounds more polite and natural"
    }}
}}"""
            
            try:
                assessment_response = model.generate_content(assessment_prompt)
                assessment_text = assessment_response.text.strip()
                
                assessment_text = assessment_text.replace('```json', '').replace('```', '').strip()
                
                if '{' in assessment_text and '}' in assessment_text:
                    json_start = assessment_text.index('{')
                    json_end = assessment_text.rindex('}') + 1
                    json_str = assessment_text[json_start:json_end]
                    assessment = json.loads(json_str)
                    
                    required_fields = ['score', 'stars', 'message', 'what_you_did_well', 'improvement_tip']
                    if all(field in assessment for field in required_fields):
                        return {"is_final": True, "assessment": assessment}
                
                assessment = {
                    "score": 85,
                    "stars": 4,
                    "message": "Great job! You completed the practice session.",
                    "what_you_did_well": "You engaged well in the conversation and showed good understanding of the topic.",
                    "improvement_tip": {
                        "what_they_said": "Your responses",
                        "better_way": "More natural phrasing with complete sentences",
                        "explanation": "Practice using full, polite sentences in conversation"
                    }
                }
                return {"is_final": True, "assessment": assessment}
                
            except Exception as e:
                print(f"Error generating assessment: {e}")
                assessment = {
                    "score": 80,
                    "stars": 4,
                    "message": "Well done! You completed the practice session.",
                    "what_you_did_well": "You participated actively and showed effort in practicing.",
                    "improvement_tip": {
                        "what_they_said": "Your conversation",
                        "better_way": "More detailed responses",
                        "explanation": "Try to elaborate more on your answers"
                    }
                }
                return {"is_final": True, "assessment": assessment}
        
        return {"is_final": False, "message": assistant_message}


@app.route('/start_session', methods=['POST'])
def start_session():
    """Initialize a new conversation session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        topic = data.get('topic', 'Ordering at a Café')
        lesson_content = data.get('lesson_content', 'Basic café ordering phrases and polite requests')
        language = data.get('language', 'english')
        
        conversations[session_id] = ConversationManager(topic, lesson_content, language)
        
        initial_response = conversations[session_id].get_response("Hello, I'm ready to practice!")
        
        audio_base64 = text_to_speech(initial_response['message'], language)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'initial_message': initial_response['message'],
            'audio': audio_base64
        })
    except Exception as e:
        print(f"Error in start_session: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Google Speech-to-Text"""
    try:
        data = request.json
        audio_content = base64.b64decode(data['audio'])
        language = data.get('language', 'english')
        
        language_map = {
            'english': 'en-US',
            'hindi': 'hi-IN',
            'kannada': 'kn-IN',
            'tamil': 'ta-IN',
            'telugu': 'te-IN',
            'malayalam': 'ml-IN',
            'bengali': 'bn-IN'
        }
        
        language_code = language_map.get(language.lower(), 'en-US')
        
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model='latest_long'
        )
        
        response = speech_client.recognize(config=config, audio=audio)
        
        if not response.results:
            return jsonify({'status': 'error', 'message': 'No speech detected'})
        
        transcript = response.results[0].alternatives[0].transcript
        confidence = response.results[0].alternatives[0].confidence
        
        return jsonify({
            'status': 'success',
            'transcript': transcript,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/send_message', methods=['POST'])
def send_message():
    """Process user message and get bot response"""
    try:
        data = request.json
        session_id = data.get('session_id')
        user_message = data.get('message')
        
        if session_id not in conversations:
            return jsonify({'status': 'error', 'message': 'Session not found'})
        
        conversation = conversations[session_id]
        response = conversation.get_response(user_message)
        
        audio_base64 = None
        if not response['is_final']:
            audio_base64 = text_to_speech(response['message'], conversation.language)
        
        return jsonify({
            'status': 'success',
            'response': response,
            'audio': audio_base64,
            'turn_count': conversation.turn_count,
            'max_turns': conversation.max_turns
        })
        
    except Exception as e:
        print(f"Error in send_message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/end_session', methods=['POST'])
def end_session():
    """End conversation session"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in conversations:
        del conversations[session_id]
    
    return jsonify({'status': 'success'})


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Language Learning Chatbot API',
        'endpoints': {
            'GET /test': 'Test interface',
            'POST /start_session': 'Initialize a new conversation session',
            'POST /transcribe_audio': 'Convert audio to text',
            'POST /send_message': 'Send user message and get bot response',
            'POST /end_session': 'End conversation session',
            'GET /health': 'Health check'
        }
    })

@app.route('/test')
def test_interface():
    """Serve test interface"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Language Learning Chatbot Test</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 600px;
            padding: 30px;
        }
        
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .topic {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .progress {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s;
        }
        
        .chat-container {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        
        .message.bot {
            align-items: flex-start;
        }
        
        .message.user {
            align-items: flex-end;
        }
        
        .message-content {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
        }
        
        .message.user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .mic-button {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-size: 24px;
            cursor: pointer;
            transition: transform 0.2s;
            flex-shrink: 0;
        }
        
        .mic-button:hover {
            transform: scale(1.1);
        }
        
        .mic-button:active {
            transform: scale(0.95);
        }
        
        .mic-button.recording {
            animation: pulse 1.5s infinite;
            background: linear-gradient(135deg, #f093fb, #f5576c);
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .text-input {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
        }
        
        .text-input:focus {
            border-color: #667eea;
        }
        
        .send-button {
            padding: 15px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
        }
        
        .send-button:hover {
            background: #5568d3;
        }
        
        .start-button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        
        .start-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .status {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }
        
        .score-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            margin-top: 20px;
        }
        
        .score {
            font-size: 48px;
            font-weight: bold;
            margin: 20px 0;
        }
        
        .stars {
            font-size: 32px;
            margin: 10px 0;
        }
        
        .feedback {
            background: white;
            color: #333;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            text-align: left;
        }
        
        .feedback h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .feedback p {
            margin-bottom: 15px;
            line-height: 1.6;
        }
        
        .improvement {
            background: #fff3cd;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #ffc107;
        }
        
        .strikethrough {
            text-decoration: line-through;
            color: #d32f2f;
        }
        
        .correct {
            color: #4caf50;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🗣️ Language Practice</h1>
        <p class="topic">Topic: <span id="topicName">Ordering at a Café</span></p>
        
        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 8px; color: #666; font-weight: 600;">Select Language:</label>
            <select id="languageSelect" style="width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 10px; font-size: 14px; background: white; cursor: pointer;">
                <option value="english">English</option>
                <option value="hindi">Hindi (हिंदी)</option>
                <option value="kannada">Kannada (ಕನ್ನಡ)</option>
                <option value="tamil">Tamil (தமிழ்)</option>
                <option value="telugu">Telugu (తెలుగు)</option>
                <option value="malayalam">Malayalam (മലയാളം)</option>
                <option value="bengali">Bengali (বাংলা)</option>
            </select>
        </div>
        
        <div class="progress-bar">
            <div class="progress" id="progressBar"></div>
        </div>
        
        <button class="start-button" id="startBtn" onclick="startSession()">Start Practice Session</button>
        
        <div class="chat-container" id="chatContainer" style="display: none;"></div>
        
        <div class="controls" id="controls" style="display: none;">
            <button class="mic-button" id="micBtn" onclick="toggleRecording()">🎤</button>
            <input type="text" class="text-input" id="textInput" placeholder="Or type your message...">
            <button class="send-button" onclick="sendTextMessage()">Send</button>
        </div>
        
        <div class="status" id="status"></div>
        
        <div id="scoreCard"></div>
    </div>

    <script>
        const API_URL = window.location.origin;
        let sessionId = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let currentLanguage = 'english';

        async function startSession() {
            sessionId = 'session_' + Date.now();
            currentLanguage = document.getElementById('languageSelect').value;
            
            const topics = {
                'english': 'Ordering at a Café',
                'hindi': 'कैफे में ऑर्डर करना',
                'kannada': 'ಕೆಫೆಯಲ್ಲಿ ಆರ್ಡರ್ ಮಾಡುವುದು',
                'tamil': 'ஒரு கஃபேயில் ஆர்டர் செய்தல்',
                'telugu': 'కేఫ్‌లో ఆర్డర్ చేయడం',
                'malayalam': 'ഒരു കഫേയിൽ ഓർഡർ ചെയ്യുക',
                'bengali': 'ক্যাফেতে অর্ডার করা'
            };
            
            const lessonContent = {
                'english': 'Basic café ordering phrases, polite requests, and common vocabulary',
                'hindi': 'कैफे में ऑर्डर करने के बुनियादी वाक्यांश, विनम्र अनुरोध, और सामान्य शब्दावली',
                'kannada': 'ಕೆಫೆಯಲ್ಲಿ ಆರ್ಡರ್ ಮಾಡಲು ಮೂಲಭೂತ ನುಡಿಗಟ್ಟುಗಳು, ವಿನಯಶೀಲ ವಿನಂತಿಗಳು ಮತ್ತು ಸಾಮಾನ್ಯ ಶಬ್ದಕೋಶ',
                'tamil': 'அடிப்படை கஃபே ஆர்டரிங் சொற்றொடர்கள், கண்ணியமான கோரிக்கைகள் மற்றும் பொதுவான சொற்களஞ்சியம்',
                'telugu': 'ప్రాథమిక కేఫ్ ఆర్డరింగ్ పదబంధాలు, మర్యాదపూర్వక అభ్యర్థనలు మరియు సాధారణ పదజాలం',
                'malayalam': 'അടിസ്ഥാന കഫേ ഓർഡറിംഗ് വാക്യങ്ങൾ, മര്യാദയുള്ള അഭ്യർത്ഥനകൾ, സാധാരണ പദാവലി',
                'bengali': 'মৌলিক ক্যাফে অর্ডারিং বাক্যাংশ, ভদ্র অনুরোধ এবং সাধারণ শব্দভাণ্ডার'
            };
            
            try {
                updateStatus('Starting session...');
                const response = await fetch(`${API_URL}/start_session`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        topic: topics[currentLanguage],
                        lesson_content: lessonContent[currentLanguage],
                        language: currentLanguage
                    })
                });
                
                const text = await response.text();
                console.log('Raw response:', text);
                
                let data;
                try {
                    data = JSON.parse(text);
                } catch (e) {
                    console.error('JSON parse error:', e);
                    console.error('Response text:', text);
                    updateStatus('Error: Invalid response from server. Check console for details.');
                    return;
                }
                
                if (data.status === 'success') {
                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('languageSelect').disabled = true;
                    document.getElementById('chatContainer').style.display = 'block';
                    document.getElementById('controls').style.display = 'flex';
                    
                    addMessage('bot', data.initial_message);
                    
                    // Play audio response
                    if (data.audio) {
                        playAudio(data.audio);
                    }
                    
                    updateStatus('Session started! Click the microphone or type your response.');
                } else {
                    updateStatus('Error: ' + (data.message || 'Unknown error'));
                }
            } catch (error) {
                console.error('Fetch error:', error);
                updateStatus('Error: ' + error.message);
            }
        }

        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm'
                });
                
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await transcribeAudio(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                document.getElementById('micBtn').classList.add('recording');
                updateStatus('Recording... Click again to stop');
                
            } catch (error) {
                updateStatus('Error accessing microphone: ' + error.message);
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                document.getElementById('micBtn').classList.remove('recording');
                updateStatus('Processing...');
            }
        }

        async function transcribeAudio(audioBlob) {
            try {
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                
                reader.onloadend = async () => {
                    const base64Audio = reader.result.split(',')[1];
                    
                    const response = await fetch(`${API_URL}/transcribe_audio`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            audio: base64Audio,
                            language: currentLanguage
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        addMessage('user', data.transcript);
                        await sendMessage(data.transcript);
                    } else {
                        updateStatus('Error: ' + data.message);
                    }
                };
            } catch (error) {
                updateStatus('Error: ' + error.message);
            }
        }

        async function sendTextMessage() {
            const input = document.getElementById('textInput');
            const message = input.value.trim();
            
            if (message) {
                addMessage('user', message);
                input.value = '';
                await sendMessage(message);
            }
        }

        async function sendMessage(message) {
            try {
                updateStatus('Thinking...');
                
                const response = await fetch(`${API_URL}/send_message`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        message: message
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    const progress = (data.turn_count / data.max_turns) * 100;
                    document.getElementById('progressBar').style.width = progress + '%';
                    
                    if (data.response.is_final) {
                        displayFinalAssessment(data.response.assessment);
                    } else {
                        addMessage('bot', data.response.message);
                        
                        // Play audio response
                        if (data.audio) {
                            playAudio(data.audio);
                        }
                        
                        updateStatus(`Turn ${data.turn_count}/${data.max_turns} - Keep going!`);
                    }
                }
            } catch (error) {
                updateStatus('Error: ' + error.message);
            }
        }

        function addMessage(type, content) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function displayFinalAssessment(assessment) {
            document.getElementById('controls').style.display = 'none';
            updateStatus('Practice session complete!');
            
            const stars = '⭐'.repeat(assessment.stars) + '☆'.repeat(5 - assessment.stars);
            
            const scoreCard = document.getElementById('scoreCard');
            scoreCard.innerHTML = `
                <div class="score-card">
                    <h2>Overall Score</h2>
                    <div class="score">${assessment.score}/100</div>
                    <div class="stars">${stars}</div>
                    <p style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; margin-top: 10px;">
                        ${assessment.message}
                    </p>
                </div>
                
                <div class="feedback">
                    <h3>👍 What you did well</h3>
                    <p>${assessment.what_you_did_well}</p>
                    
                    <h3>💡 Improvement Tip</h3>
                    <div class="improvement">
                        <p><strong>You said:</strong><br>
                        <span class="strikethrough">"${assessment.improvement_tip.what_they_said}"</span></p>
                        
                        <p><strong>Better way:</strong><br>
                        <span class="correct">"${assessment.improvement_tip.better_way}"</span></p>
                        
                        <p><strong>Why:</strong> ${assessment.improvement_tip.explanation}</p>
                    </div>
                </div>
            `;
        }

        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }

        function playAudio(audioBase64) {
            try {
                const audio = new Audio('data:audio/mp3;base64,' + audioBase64);
                audio.play().catch(err => {
                    console.error('Error playing audio:', err);
                });
            } catch (error) {
                console.error('Error creating audio:', error);
            }
        }

        document.getElementById('textInput')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendTextMessage();
            }
        });
    </script>
</body>
</html>
    '''

@app.route('/test_gemini', methods=['GET'])
def test_gemini():
    """Test Gemini API connection"""
    try:
        test_response = model.generate_content("Say 'Hello! The API is working.'")
        return jsonify({
            'status': 'success',
            'message': test_response.text
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)