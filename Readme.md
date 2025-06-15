# Enhanced Faculty Collaboration Predictor

An advanced AI-powered system for predicting optimal faculty collaboration teams for grant proposals using machine learning, knowledge graphs, and natural language processing.

## 🚀 Features

### Core Capabilities
- **Multi-Modal Text Analysis**: Combines BERT embeddings, sentence transformers, and advanced keyword extraction
- **Knowledge Graph Integration**: Builds RDF-based semantic networks of faculty expertise
- **Machine Learning Models**: Trains on historical collaboration data for optimal team prediction
- **Interactive Visualizations**: Creates network graphs and collaboration maps
- **Comprehensive Reporting**: Generates detailed analysis reports in multiple formats

### Advanced Analytics
- **Team Diversity Analysis**: Evaluates department, methodology, and topical diversity
- **Collaboration History**: Analyzes existing faculty partnerships and success patterns
- **Expertise Coverage**: Maps team capabilities against grant requirements
- **Team Modification Suggestions**: Recommends additions or replacements to existing teams

## 📋 Requirements

### Core Dependencies
```
python >= 3.8
torch >= 1.9.0
transformers >= 4.20.0
scikit-learn >= 1.0.0
spacy >= 3.4.0
sentence-transformers >= 2.2.0
networkx >= 2.8
gensim >= 4.2.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
rdflib >= 6.2.0
```

### Optional Dependencies
```
# For advanced keyword extraction
keybert >= 0.7.0
yake >= 0.4.8

# For interactive visualizations
pyvis >= 0.2.1
plotly >= 5.10.0
wordcloud >= 1.8.0

# For PDF generation
reportlab >= 3.6.0

# For PDF text extraction
pdfminer.six >= 20220319

# For enhanced NLP
spacy-dbpedia-spotlight >= 0.2.2
```

### spaCy Language Model
```bash
python -m spacy download en_core_web_lg
```

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/faculty-collaboration-predictor.git
cd faculty-collaboration-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_lg
```

5. **Optional: Install advanced features**
```bash
pip install keybert yake pyvis plotly wordcloud reportlab pdfminer.six
```

## 🚦 Quick Start

### Basic Usage

```python
from faculty_collaboration_predictor import EnhancedFacultyCollaborationPredictor

# Initialize the predictor
predictor = EnhancedFacultyCollaborationPredictor()

# Load faculty biographical data
predictor.load_faculty_bios("data/faculty_bios/")

# Optional: Load research papers for better topic modeling
predictor.load_research_papers("data/research_papers/")

# Define your grant proposal
grant_proposal = """
We propose to develop a novel machine learning approach for analyzing 
genomic data to identify biomarkers for cancer diagnosis. This interdisciplinary 
project requires expertise in computational biology, machine learning, 
statistical analysis, and clinical research.
"""

# Predict optimal collaboration teams
results = predictor.predict_collaborations(
    grant_proposal,
    top_n=5,
    team_sizes=[2, 3, 4],
    use_ml=True
)

# Display results
for i, team in enumerate(results):
    print(f"Team {i+1}: {', '.join(team['faculty_names'])}")
    print(f"Score: {team['combined_score']:.3f}")
    print(f"Synergies: {'; '.join(team['synergy_reasons'])}")
    print()
```

### Advanced Features

```python
# Train ML model with historical data
predictor.train_ml_model("data/historical_grants.json")

# Generate comprehensive report
report = predictor.generate_detailed_report(results, output_format='pdf')

# Visualize collaboration network
predictor.visualize_collaboration_network(
    "collaboration_network.html", 
    interactive=True,
    color_by_department=True
)

# Export knowledge graph
predictor.export_knowledge_graph(
    format='turtle', 
    filename='faculty_knowledge_graph.ttl'
)

# Get faculty statistics
stats = predictor.get_faculty_statistics()
print(f"Total faculty: {stats['total_faculty']}")
print(f"Departments: {list(stats['departments'].keys())}")
```

## 📁 Data Structure

### Faculty Bio Files
Place faculty biographical files in the specified directory:
```
data/
├── faculty_bios/
│   ├── faculty_001.txt
│   ├── faculty_002.pdf
│   └── ...
├── research_papers/
│   ├── paper_001.pdf
│   ├── paper_002.txt
│   └── ...
└── historical_grants.json
```

### Faculty Bio Format
Each bio file should contain:
- Faculty name
- Department affiliation
- Research interests
- Publications
- Grant history
- Methodological expertise

Example:
```
Dr. Jane Smith
Department of Computer Science

Research Interests: Machine learning, natural language processing, 
computational biology, and data mining.

Dr. Smith has published over 50 papers in top-tier conferences...
```

### Historical Training Data Format
```json
[
  {
    "grant_id": "NSF-2021-001",
    "team": ["faculty_001", "faculty_003", "faculty_007"],
    "proposal_text": "Grant proposal text...",
    "success_score": 0.85,
    "outcome": "funded",
    "year": 2021
  }
]
```

## 🎯 Key Methods

### Core Prediction Methods
- `predict_collaborations()`: Main prediction method with ML integration
- `predict_collaborations_with_ml()`: ML-enhanced team recommendation
- `analyze_grant_proposal_enhanced()`: Comprehensive proposal analysis

### Analysis Methods
- `_analyze_team_diversity()`: Evaluate team diversity metrics
- `_analyze_collaboration_network()`: Assess existing collaborations
- `_analyze_expertise_coverage()`: Map expertise to requirements
- `suggest_team_modifications()`: Recommend team improvements

### Visualization Methods
- `visualize_collaboration_network()`: Create network visualizations
- `generate_detailed_report()`: Comprehensive reporting
- `export_knowledge_graph()`: Export semantic knowledge

### Utility Methods
- `load_faculty_bios()`: Process faculty biographical data
- `load_research_papers()`: Analyze research publications
- `train_ml_model()`: Train collaboration prediction models
- `get_faculty_statistics()`: Generate system statistics

## ⚙️ Configuration

### Optimal Weights
Customize the scoring weights in the predictor:
```python
predictor.optimal_weights = {
    'embedding_weight': 0.3,      # Semantic similarity weight
    'keyword_weight': 0.2,        # Keyword matching weight
    'entity_weight': 0.15,        # Entity recognition weight
    'topic_weight': 0.15,         # Topic modeling weight
    'knowledge_graph_weight': 0.2, # Knowledge graph weight
    'collaboration_weight': 0.1,  # Historical collaboration weight
    'diversity_weight': 0.15      # Team diversity weight
}
```

### Model Parameters
```python
# Topic modeling
predictor.num_topics = 30

# Team size preferences
team_sizes = [2, 3, 4, 5]

# Number of recommendations
top_n = 10
```

## 📊 Evaluation Metrics

The system uses multiple scoring mechanisms:

### Individual Faculty Scores
- **Semantic Similarity**: BERT and sentence transformer embeddings
- **Keyword Matching**: Advanced keyword extraction and matching
- **Entity Recognition**: Named entity alignment
- **Topic Modeling**: LDA-based topic distribution similarity
- **Knowledge Graph**: Semantic relationship scoring

### Team Combination Scores
- **Base Score**: Average individual faculty scores
- **Collaboration Bonus**: Historical partnership strength
- **Diversity Score**: Interdisciplinary team composition
- **Complementarity Score**: Unique expertise contributions
- **Coverage Score**: Grant requirement fulfillment

## 🔬 Machine Learning Models

### Supported Algorithms
- **Random Forest**: Ensemble learning for collaboration prediction
- **Gradient Boosting**: Advanced ensemble with feature importance
- **Custom Ensemble**: Combines multiple model predictions

### Training Features
- Team size and composition
- Department diversity metrics
- Historical collaboration patterns
- Semantic similarity distributions
- Research method overlap
- Publication timeline analysis

## 📈 Visualization Options

### Network Graphs
- **Static plots**: Matplotlib/Seaborn visualizations
- **Interactive graphs**: Pyvis-powered network exploration
- **Knowledge graphs**: RDF triple visualizations

### Report Formats
- **Dictionary**: Python data structures
- **JSON**: Structured data export
- **PDF**: Professional report generation
- **HTML**: Interactive web reports

## 🛠️ Advanced Configuration

### GPU Acceleration
```python
# Enable GPU for faster processing
predictor = EnhancedFacultyCollaborationPredictor(
    embedding_model="allenai/scibert_scivocab_uncased",
    use_gpu=True
)
```

### Custom Embedding Models
```python
# Use domain-specific models
predictor = EnhancedFacultyCollaborationPredictor(
    embedding_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
)
```

### DBpedia Integration
```bash
# Enable semantic entity linking
export USE_DBPEDIA_SPOTLIGHT=true
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch sizes for large datasets
   - Use CPU-only mode if GPU memory is limited
   - Process faculty bios in smaller chunks

2. **Missing Dependencies**
   - Install optional packages as needed
   - Check spaCy model installation
   - Verify PyTorch CUDA compatibility

3. **Data Format Issues**
   - Ensure proper text encoding (UTF-8)
   - Check file extensions (.txt, .pdf)
   - Validate JSON structure for training data

### Performance Optimization

```python
# For large datasets
predictor = EnhancedFacultyCollaborationPredictor()
predictor.num_topics = 20  # Reduce for faster processing
results = predictor.predict_collaborations(
    grant_proposal,
    top_n=3,  # Fewer recommendations
    team_sizes=[2, 3]  # Smaller teams
)
```

## 📝 Example Applications

### Academic Research Institutions
- Grant proposal team formation
- Cross-departmental collaboration identification
- Research center composition planning

### Industry R&D Teams
- Project team optimization
- Expertise gap analysis
- Collaboration network development

### Funding Agencies
- Proposal evaluation assistance
- Team composition assessment
- Success prediction modeling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{faculty_collaboration_predictor,
  title={Enhanced Faculty Collaboration Predictor},
  author={Sameer Mankotia},
  year={2025},
  url={https://github.com/your-repo/faculty-collaboration-predictor}
}
```

## 🔄 Version History

- **v1.0.0**: Initial release with basic collaboration prediction
- **v2.0.0**: Added machine learning and knowledge graph integration
- **v2.1.0**: Enhanced visualization and reporting capabilities
- **v2.2.0**: Advanced team modification suggestions and optimization

---

**Built with ❤️ for the academic research community by Sameer Mankotia**