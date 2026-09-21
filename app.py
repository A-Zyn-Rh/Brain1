import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

# 🔑 API Key Gemini Kamu
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Inisialisasi Client Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY))

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        system_prompt = data.get('system', '')

        # Format pesan untuk Gemini API
        formatted_contents = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            content_text = msg.get("content", "")
            
            formatted_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content_text)]
                )
            )

        # Konfigurasi permintaan
        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            max_output_tokens=1000,
            temperature=0.7
        )

        # Panggil model Gemini
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=formatted_contents,
            config=config
        )

        total_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            total_tokens = (response.usage_metadata.prompt_token_count or 0) + (response.usage_metadata.candidates_token_count or 0)

        return jsonify({
            "text": response.text,
            "usage": total_tokens
        })

    except Exception as e:
        print("Error Gemini API:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Server Python Gemini berjalan di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)