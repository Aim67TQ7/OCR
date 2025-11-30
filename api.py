# api.py (ONE FILE - put this with your agent files)
from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import base64

from text_agent import TextExtractionAgent
from table_agent import ProductionTableAgent
from ocr_agent import OCRProcessingAgent

app = Flask(__name__)
CORS(app)  # Allow requests from your React app

# Initialize agents
text_agent = TextExtractionAgent()
table_agent = ProductionTableAgent()
ocr_agent = OCRProcessingAgent()

@app.route('/parse', methods=['POST'])
def parse_document():
    try:
        # Get file from request
        if 'file' in request.files:
            file = request.files['file']
            file_bytes = file.read()
            filename = file.filename
        else:
            # Handle base64 from n8n
            data = request.json
            file_bytes = base64.b64decode(data['file_base64'])
            filename = data.get('filename', 'document.pdf')
        
        # Run extraction (sync wrapper for async code)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        text_result = loop.run_until_complete(
            text_agent.extract_text(file_bytes, filename)
        )
        
        table_result = loop.run_until_complete(
            table_agent.extract_tables(file_bytes, filename)
        )
        
        # Return combined results
        return jsonify({
            'success': True,
            'filename': filename,
            'text': text_result.text,
            'tables': table_result.tables,
            'confidence_score': text_result.confidence_score,
            'processing_time_ms': text_result.processing_time_ms,
            'metadata': text_result.metadata
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Step 2: Deploy to Render.com (FREE - Easiest)

1. **Put all files in a folder:**
```
parsing-service/
├── api.py
├── text_agent.py
├── table_agent.py
├── ocr_agent.py
├── requirements.txt
└── render.yaml
