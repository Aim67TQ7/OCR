from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import asyncio

# Import your agents
# ocr_agent import OCRProcessingAgent
#from text_agent import TextExtractionAgent
#from table_agent import ProductionTableAgent

app = Flask(__name__)
CORS(app)

# Initialize agents
ocr_agent = OCRProcessingAgent()
text_agent = TextExtractionAgent()
table_agent = ProductionTableAgent()

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'service': 'document-parsing'})

@app.route('/parse', methods=['POST'])
def parse_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_bytes = file.read()
        filename = file.filename
        
        # Run async extraction
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Try text extraction first
        text_result = loop.run_until_complete(
            text_agent.extract_text(file_bytes, filename)
        )
        
        # Try table extraction
        table_result = loop.run_until_complete(
            table_agent.extract_tables(file_bytes, filename)
        )
        
        return jsonify({
            'success': True,
            'filename': filename,
            'text': text_result.text,
            'tables': table_result.tables[:5],  # Limit to first 5 tables
            'confidence': text_result.confidence_score,
            'processing_time_ms': text_result.processing_time_ms
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
