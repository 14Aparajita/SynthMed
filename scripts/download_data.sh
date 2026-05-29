#!/bin/bash
# Download and prepare mini DR dataset for SynthMed

set -e

echo "Setting up SynthMed data directory..."

# Create directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/knowledge_base

echo "Creating sample knowledge base documents..."

# Create knowledge base file
python -c "
import json
from pathlib import Path

documents = [
    {'text': 'Diabetic retinopathy (DR) is a microvascular complication of diabetes affecting retinal blood vessels.'},
    {'text': 'Microaneurysms are the earliest visible sign of diabetic retinopathy, appearing as small red dots.'},
    {'text': 'Hard exudates are yellow-white lipoprotein deposits often forming circinate patterns.'},
    {'text': 'Proliferative DR involves neovascularization and can lead to vitreous hemorrhage.'},
    {'text': 'Diabetic macular edema is the leading cause of vision loss in DR patients.'},
    {'text': 'Regular screening enables early detection and treatment of diabetic retinopathy.'},
    {'text': 'Severe NPDR shows venous beading, IRMA, and extensive hemorrhages.'},
    {'text': 'Anti-VEGF therapy is first-line treatment for diabetic macular edema.'},
    {'text': 'Cotton wool spots indicate retinal ischemia from microvascular occlusion.'},
    {'text': 'DR grading: 0=no DR, 1=mild NPDR, 2=moderate, 3=severe, 4=PDR.'},
]

with open('data/knowledge_base/pubmed_abstracts.jsonl', 'w') as f:
    for doc in documents:
        f.write(json.dumps(doc) + '\n')

print(f'Created {len(documents)} knowledge base documents')
"

echo "Data setup complete!"
echo "Note: Place your retinal images in data/raw/ for real data processing."
echo "The system will generate synthetic placeholder data if no images are found."