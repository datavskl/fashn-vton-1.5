## Installation Guide on L40s or L4
git clone https://github.com/fashn-AI/fashn-vton-1.5.git
cd fashn-vton-1.5

## Install dependencies
pip install -e .
## Or
pip install -r requirements.txt

## Download Weights
python scripts/download_weights.py --weights-dir ./weights
