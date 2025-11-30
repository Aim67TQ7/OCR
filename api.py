from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Only import what we have installed
# from ocr_agent import OCRProcessingAgent  # COMMENT OUT
# from text_agent import TextExtractionAgent  # COMMENT OUT
# from table_agent import ProductionTableAgent  # COMMENT OUT

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'service': 'parsing-api'})

@app.route('/parse', methods=['POST'])
def parse_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_bytes = file.read()
        
        # Basic file info for now
        return jsonify({
            'success': True,
            'filename': file.filename,
            'file_size': len(file_bytes),
            'message': 'File received - parsing agents will be added next'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
