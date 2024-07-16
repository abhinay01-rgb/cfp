from flask import Flask, request, render_template, jsonify
import boto3
import fitz  # PyMuPDF
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'proposals'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to analyze and group texts
def analyze_pdf_files(text):
    try:
        # Initialize Boto3 client for Bedrock
        bedrock = boto3.client('bedrock-runtime')
        
        # Construct a concise prompt to categorize the text
        prompt_data = f'''Analyze the following text in Human-Computer Interaction (HCI):\n\n{text}\n\n
        Categorize this text into relevant technology domains and find the title of the proposal.
        Highlight the key technologies such as artificial intelligence, machine learning, computer-vision, augmented-reality, virtual-reality,
        mixed-reality, user-interface used and specify whether it is research-oriented or product-oriented.
        Evaluate the uniqueness, speciality, originality, or novelty of this research topic {text} and provide reasons for your assessment.
        Assess its relevance to Human-Computer Interaction and its alignment with end-user needs.
        Predict the potential impact of the proposed solution.
        Determine whether this research could lead to a tangible product.
       
        Write the answer in the following format:
        title of proposal: [title of proposal] \n
        list of key technologies: [list of key technologies] \n 
        novelty assessment: [novelty assessment] \n
        relevance assessment to HCI: [relevance assessment to HCI]\n
        impact assessment: [impact assessment]\n
        potential for productization: [potential for productization]\n
        concise summary: [concise summary]\n'''

        # Prepare and send request to Bedrock
        payload = {
            "prompt": prompt_data,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048  # Adjust max tokens to allow for a longer response
        }
        body = json.dumps(payload)
    
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        
        response = bedrock.invoke_model(
            modelId=model_id,
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        
        # Read and decode the response body
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        # Extract the text part from the response
        if "outputs" in response_body and len(response_body["outputs"]) > 0 and "text" in response_body["outputs"][0]:
            print(response_body["outputs"][0]["text"])
            return response_body["outputs"][0]["text"]
        else:
            return "No text found in response"
    
    except Exception as e:
        return f"Error processing texts: {e}"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'proposal' not in request.files:
        return 'No file part', 400
    
    file = request.files['proposal']
    if file.filename == '':
        return 'No selected file', 400
    
    if file and file.filename.endswith('.pdf'):
        # Remove existing files in the folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Save the new file
        file.save(os.path.join(UPLOAD_FOLDER, 'proposal.pdf'))
        return 'File uploaded successfully', 200
    
    return 'Invalid file type', 400

@app.route('/analyze', methods=['POST'])
def analyze():
    folder_path = UPLOAD_FOLDER
    if not os.path.exists(folder_path):
        return "Folder 'proposals' not found", 400

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        return "No PDF files found in the 'proposals' folder", 400

    results = []
    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(folder_path, pdf_file)
        text = extract_text_from_pdf(pdf_file_path)
        analysis_result = analyze_pdf_files(text)
        parsed_result = parse_analysis_result(analysis_result)
        results.append({
            'file': pdf_file,
            'result': parsed_result
        })

    return jsonify(results)

def parse_analysis_result(analysis_result):
    # Extract relevant parts from the analysis result text
    sections = analysis_result.split('\n')
    result_dict = {
        "title of proposal": "",
        "list of key technologies": "",
        "novelty assessment": "",
        "relevance assessment to HCI": "",
        "impact assessment": "",
        "potential for productization": "",
        "concise summary": ""
    }
    
    current_section = None
    for line in sections:
        if line.startswith("title of proposal:"):
            current_section = "title of proposal"
        elif line.startswith("list of key technologies:"):
            current_section = "list of key technologies"
        elif line.startswith("novelty assessment:"):
            current_section = "novelty assessment"
        elif line.startswith("relevance assessment to HCI:"):
            current_section = "relevance assessment to HCI"
        elif line.startswith("impact assessment:"):
            current_section = "impact assessment"
        elif line.startswith("potential for productization:"):
            current_section = "potential for productization"
        elif line.startswith("concise summary:"):
            current_section = "concise summary"
        
        if current_section and line.startswith(current_section):
            result_dict[current_section] = line[len(current_section) + 1:].strip()
        elif current_section:
            result_dict[current_section] += " " + line.strip()
    
    return result_dict

if __name__ == "__main__":
    app.run(debug=True)
