# Enhanced Faculty Collaboration Prediction System with ML and Knowledge Graph
# Complete system with machine learning, knowledge graphs, and advanced features

import os
import re
import pandas as pd
import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
from gensim.models import LdaModel
from pdfminer.high_level import extract_text
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For knowledge graph
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("pyvis not available - interactive network visualization will be limited")

import rdflib
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import FOAF, RDFS, OWL

# For advanced NLP
from transformers import pipeline
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("KeyBERT not available - using basic keyword extraction")

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("YAKE not available - using basic keyword extraction")

import nltk
try:
    from nltk.corpus import wordnet
    nltk.download('wordnet', quiet=True)
except:
    pass

# For visualization
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud

# For PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not available - PDF generation will be limited")

class EnhancedFacultyCollaborationPredictor:
    def __init__(self, embedding_model="allenai/scibert_scivocab_uncased", use_gpu=True):
        """
        Initialize the enhanced collaboration prediction system with ML and knowledge graph capabilities
        
        Parameters:
        embedding_model (str): The sentence transformer model to use for embeddings
        use_gpu (bool): Whether to use GPU for computation
        """
        # Load advanced NLP models
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Using device: {self.device}")
        
        # Load spaCy with entity linking
        self.nlp = spacy.load("en_core_web_lg")
        self.use_dbpedia = False
        
        # Try to add DBpedia Spotlight, but make it optional
        try:
            # Only try to import if explicitly requested
            if os.environ.get('USE_DBPEDIA_SPOTLIGHT', 'false').lower() == 'true':
                import spacy_dbpedia_spotlight
                self.nlp.add_pipe('dbpedia_spotlight', config={'confidence': 0.5, 'timeout': 5})
                self.use_dbpedia = True
                print("DBpedia Spotlight enabled")
        except Exception as e:
            print(f"DBpedia Spotlight not available: {str(e)}")
        
        # Load BERT for contextual embeddings
        self.bert_tokenizer = BertTokenizer.from_pretrained(embedding_model)
        self.bert_model = BertModel.from_pretrained(embedding_model).to(self.device)
        self.bert_model.eval()
        
        # Sentence transformer for general embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # KeyBERT for advanced keyword extraction
        if KEYBERT_AVAILABLE:
            self.keybert = KeyBERT(model=self.sentence_model)
        else:
            self.keybert = None
        
        # Initialize knowledge graph
        self.knowledge_graph = rdflib.Graph()
        self.init_knowledge_graph_namespaces()
        
        # Storage for processed data
        self.faculty_profiles = {}
        self.research_papers = {}
        self.collaboration_graph = nx.Graph()
        self.grant_history = []  # For ML training
        
        # Topic modeling
        self.dictionary = None
        self.lda_model = None
        self.num_topics = 30  # Increased for better granularity
        
        # Machine learning models
        self.ml_model = None
        self.feature_scaler = StandardScaler()
        self.feature_names = []
        
        # Hyperparameters (will be optimized)
        self.optimal_weights = {
            'embedding_weight': 0.3,
            'keyword_weight': 0.2,
            'entity_weight': 0.15,
            'topic_weight': 0.15,
            'knowledge_graph_weight': 0.2,
            'collaboration_weight': 0.1,
            'diversity_weight': 0.15
        }
        
    def init_knowledge_graph_namespaces(self):
        """Initialize RDF namespaces for the knowledge graph"""
        self.FACULTY = Namespace("http://faculty.edu/")
        self.RESEARCH = Namespace("http://research.edu/")
        self.COLLAB = Namespace("http://collaboration.edu/")
        
        self.knowledge_graph.bind("faculty", self.FACULTY)
        self.knowledge_graph.bind("research", self.RESEARCH)
        self.knowledge_graph.bind("collab", self.COLLAB)
        self.knowledge_graph.bind("foaf", FOAF)
        self.knowledge_graph.bind("rdfs", RDFS)
        
    def extract_contextual_embeddings(self, text, max_length=512):
        """
        Extract contextual embeddings using BERT
        
        Parameters:
        text (str): Input text
        max_length (int): Maximum sequence length
        
        Returns:
        numpy.ndarray: Contextual embeddings
        """
        # Tokenize and encode text
        inputs = self.bert_tokenizer(text, return_tensors="pt", 
                                    max_length=max_length, 
                                    truncation=True, 
                                    padding=True).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use CLS token embedding as document representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def extract_advanced_keywords(self, text, num_keywords=20):
        """
        Extract keywords using multiple methods and combine results
        
        Parameters:
        text (str): Input text
        num_keywords (int): Number of keywords to extract
        
        Returns:
        list: Extracted keywords with scores
        """
        all_keywords = {}
        
        # KeyBERT extraction if available
        if KEYBERT_AVAILABLE and self.keybert:
            keybert_keywords = self.keybert.extract_keywords(text, 
                                                            keyphrase_ngram_range=(1, 3), 
                                                            stop_words='english',
                                                            top_n=num_keywords)
            for kw, score in keybert_keywords:
                all_keywords[kw.lower()] = score
        
        # YAKE extraction if available
        if YAKE_AVAILABLE:
            yake_extractor = yake.KeywordExtractor(lan="en", 
                                            n=3, 
                                            dedupLim=0.7, 
                                            top=num_keywords, 
                                            features=None)
            yake_keywords = yake_extractor.extract_keywords(text)
            
            for kw, score in yake_keywords:
                if kw.lower() in all_keywords:
                    all_keywords[kw.lower()] = (all_keywords[kw.lower()] + (1-score)) / 2
                else:
                    all_keywords[kw.lower()] = 1 - score
        
        # Fallback to basic extraction if neither is available
        if not all_keywords:
            doc = self.nlp(text)
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if 2 <= len(chunk.text.split()) <= 3:
                    all_keywords[chunk.text.lower()] = 0.5
            
            # Extract important terms using TF-IDF
            if len(all_keywords) < num_keywords:
                tfidf = TfidfVectorizer(max_features=num_keywords, ngram_range=(1, 3), stop_words='english')
                try:
                    tfidf_matrix = tfidf.fit_transform([text])
                    feature_names = tfidf.get_feature_names_out()
                    scores = tfidf_matrix.toarray()[0]
                    
                    for i, term in enumerate(feature_names):
                        if scores[i] > 0 and term not in all_keywords:
                            all_keywords[term] = scores[i]
                except:
                    pass
        
        # Sort by score and return
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:num_keywords]
    
    def build_faculty_knowledge_graph(self, faculty_id, profile):
        """
        Add faculty information to the knowledge graph
        
        Parameters:
        faculty_id (str): Faculty identifier
        profile (dict): Faculty profile data
        """
        faculty_uri = URIRef(f"{self.FACULTY}{faculty_id}")
        
        # Add basic information
        self.knowledge_graph.add((faculty_uri, RDF.type, self.FACULTY.Faculty))
        self.knowledge_graph.add((faculty_uri, FOAF.name, Literal(profile['name'])))
        self.knowledge_graph.add((faculty_uri, self.FACULTY.department, Literal(profile['department'])))
        
        # Add research interests
        for interest in profile.get('research_interests_list', []):
            interest_uri = URIRef(f"{self.RESEARCH}{interest.replace(' ', '_')}")
            self.knowledge_graph.add((faculty_uri, self.RESEARCH.hasInterest, interest_uri))
            self.knowledge_graph.add((interest_uri, RDFS.label, Literal(interest)))
        
        # Add keywords as research topics
        for keyword, score in profile.get('advanced_keywords', [])[:10]:
            topic_uri = URIRef(f"{self.RESEARCH}topic_{keyword.replace(' ', '_')}")
            self.knowledge_graph.add((faculty_uri, self.RESEARCH.researchTopic, topic_uri))
            self.knowledge_graph.add((topic_uri, RDFS.label, Literal(keyword)))
            self.knowledge_graph.add((topic_uri, self.RESEARCH.relevanceScore, Literal(score)))
        
        # Add entities
        for entity_type, entities in profile.get('entities', {}).items():
            for entity in entities:
                # Handle both string and dict formats
                if isinstance(entity, dict):
                    entity_text = entity.get('text', '')
                else:
                    entity_text = str(entity)
                
                if entity_text:  # Only process non-empty entities
                    entity_uri = URIRef(f"{self.RESEARCH}entity_{entity_text.replace(' ', '_')}")
                    self.knowledge_graph.add((faculty_uri, self.RESEARCH.relatedTo, entity_uri))
                    self.knowledge_graph.add((entity_uri, RDF.type, Literal(entity_type)))
                    self.knowledge_graph.add((entity_uri, RDFS.label, Literal(entity_text)))
    
    def query_knowledge_graph(self, grant_keywords, grant_entities):
        """
        Query the knowledge graph to find relevant faculty
        
        Parameters:
        grant_keywords (list): Keywords from grant proposal
        grant_entities (dict): Entities from grant proposal
        
        Returns:
        dict: Faculty relevance scores based on knowledge graph
        """
        faculty_scores = {}
        
        # SPARQL-like queries (simplified for demonstration)
        for faculty_uri in self.knowledge_graph.subjects(RDF.type, self.FACULTY.Faculty):
            faculty_id = str(faculty_uri).split('/')[-1]
            score = 0
            
            # Check keyword matches
            for keyword, _ in grant_keywords:
                topic_uri = URIRef(f"{self.RESEARCH}topic_{keyword.replace(' ', '_')}")
                if (faculty_uri, self.RESEARCH.researchTopic, topic_uri) in self.knowledge_graph:
                    score += 1
            
            # Check entity matches
            for entity_type, entities in grant_entities.items():
                for entity in entities:
                    entity_uri = URIRef(f"{self.RESEARCH}entity_{entity.replace(' ', '_')}")
                    if (faculty_uri, self.RESEARCH.relatedTo, entity_uri) in self.knowledge_graph:
                        score += 0.5
            
            if score > 0:
                faculty_scores[faculty_id] = score
        
        # Normalize scores
        max_score = max(faculty_scores.values()) if faculty_scores else 1
        for faculty_id in faculty_scores:
            faculty_scores[faculty_id] /= max_score
        
        return faculty_scores
    
    def load_faculty_bios(self, directory_path):
        """
        Load and process faculty bio files with enhanced NLP and knowledge graph construction
        
        Parameters:
        directory_path (str): Path to directory containing faculty bio files
        """
        print(f"Loading faculty bios from {directory_path} with enhanced processing...")
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt') or filename.endswith('.pdf'):
                faculty_id = os.path.splitext(filename)[0]
                file_path = os.path.join(directory_path, filename)
                
                # Extract text
                if filename.endswith('.pdf'):
                    content = extract_text(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                
                # Basic info extraction
                name = self._extract_name(content)
                department = self._extract_department(content)
                research_interests = self._extract_research_interests(content)
                
                # Process with spaCy
                doc = self.nlp(content)
                
                # Extract contextual embeddings
                bert_embeddings = self.extract_contextual_embeddings(content)
                sentence_embeddings = self.sentence_model.encode([content])[0]
                
                # Extract advanced keywords
                advanced_keywords = self.extract_advanced_keywords(content)
                
                # Extract entities with DBpedia links (if available)
                entities = self._extract_entities_with_links(doc)
                
                # Extract research interests as a list
                research_interests_list = self._extract_research_interests_list(content, doc)
                
                # Store faculty profile
                self.faculty_profiles[faculty_id] = {
                    'name': name,
                    'department': department,
                    'research_interests': research_interests,
                    'research_interests_list': research_interests_list,
                    'bio_text': content,
                    'bert_embeddings': bert_embeddings,
                    'sentence_embeddings': sentence_embeddings,
                    'keywords': self._extract_keywords(doc),
                    'advanced_keywords': advanced_keywords,
                    'entities': entities,
                    'publication_years': self._extract_publication_years(content),
                    'research_methods': self._extract_research_methods(content),
                    'grants_mentioned': self._extract_grant_info(content)
                }
                
                # Add to collaboration graph
                self.collaboration_graph.add_node(faculty_id, 
                                                type='faculty',
                                                name=name,
                                                department=department)
                
                # Build knowledge graph
                self.build_faculty_knowledge_graph(faculty_id, self.faculty_profiles[faculty_id])
                
                print(f"Processed bio for faculty: {name} with enhanced features")
    
    def _extract_entities_with_links(self, doc):
        """Extract entities with potential DBpedia links"""
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            entity_info = {
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
            # Only try to get DBpedia URI if it's enabled and available
            if self.use_dbpedia and hasattr(ent, 'kb_id_') and ent.kb_id_:
                entity_info['dbpedia_uri'] = ent.kb_id_
            
            entities[ent.label_].append(entity_info)
        
        return entities
    
    def _extract_research_interests_list(self, text, doc):
        """Extract research interests as a structured list"""
        interests = []
        
        # Pattern matching for research interests sections
        patterns = [
            r"research interests?:?\s*([^.]+)",
            r"interested in:?\s*([^.]+)",
            r"focus(?:es)? on:?\s*([^.]+)",
            r"specializ(?:es|ation) in:?\s*([^.]+)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                interest_text = match.group(1)
                # Split by commas, semicolons, or 'and'
                parts = re.split(r'[,;]|\band\b', interest_text)
                for part in parts:
                    cleaned = part.strip()
                    if len(cleaned) > 3 and len(cleaned) < 100:
                        interests.append(cleaned)
        
        # Also extract from noun phrases in relevant sections
        for chunk in doc.noun_chunks:
            if any(keyword in chunk.text.lower() for keyword in ['research', 'study', 'analysis', 'investigation']):
                if len(chunk.text) > 5 and len(chunk.text) < 50:
                    interests.append(chunk.text)
        
        return list(set(interests))  # Remove duplicates
    
    def _extract_publication_years(self, text):
        """Extract years mentioned in publications"""
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text)
        return [int(year) for year in years]
    
    def _extract_research_methods(self, text):
        """Extract research methods and techniques"""
        methods = []
        method_keywords = [
            'machine learning', 'deep learning', 'neural network', 'statistical analysis',
            'qualitative research', 'quantitative research', 'experimental design',
            'computational modeling', 'simulation', 'data mining', 'natural language processing',
            'computer vision', 'bioinformatics', 'proteomics', 'genomics'
        ]
        
        text_lower = text.lower()
        for method in method_keywords:
            if method in text_lower:
                methods.append(method)
        
        return methods
    
    def _extract_grant_info(self, text):
        """Extract grant information from text"""
        grants = []
        grant_patterns = [
            r'(NSF|NIH|DOE|DARPA|NASA)\s+(?:grant|award)',
            r'grant\s+(?:number|#)?\s*([A-Z0-9-]+)',
            r'funded by\s+([^.]+)'
        ]
        
        for pattern in grant_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                grants.append(match.group(0))
        
        return grants
    
    def train_ml_model(self, training_data_path=None):
        """
        Train machine learning model for collaboration prediction
        
        Parameters:
        training_data_path (str): Path to historical grant team data
        """
        print("Training ML model for collaboration prediction...")
        
        if training_data_path and os.path.exists(training_data_path):
            # Load historical data
            with open(training_data_path, 'r') as f:
                historical_data = json.load(f)
            
            # Extract features and labels
            X, y = self._prepare_ml_training_data(historical_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train ensemble model
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                # Hyperparameter tuning
                if name == 'rf':
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10]
                    }
                else:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'max_depth': [3, 5, 7]
                    }
                
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                
                score = grid_search.score(X_test_scaled, y_test)
                print(f"{name.upper()} model R² score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = grid_search.best_estimator_
            
            self.ml_model = best_model
            
            # Extract feature importances for weight optimization
            if hasattr(best_model, 'feature_importances_'):
                self._optimize_weights_from_ml(best_model.feature_importances_)
        
        else:
            print("No historical data available. Using default weights.")
            # Initialize a simple model with synthetic data
            self._train_synthetic_model()
    
    def _prepare_ml_training_data(self, historical_data):
        """Prepare training data for ML model"""
        X = []
        y = []
        
        for grant in historical_data:
            team = grant['team']
            success_score = grant['success_score']  # 0-1 scale
            
            # Extract team features
            features = self._extract_team_features(team, grant['proposal_text'])
            X.append(features)
            y.append(success_score)
        
        self.feature_names = self._get_feature_names()
        return np.array(X), np.array(y)
    
    def _extract_team_features(self, team, proposal_text):
        """Extract features for a team given a proposal"""
        features = []
        
        # Team size
        features.append(len(team))
        
        # Department diversity
        departments = [self.faculty_profiles[fid]['department'] for fid in team if fid in self.faculty_profiles]
        features.append(len(set(departments)) / len(departments) if departments else 0)
        
        # Average similarity scores
        proposal_embedding = self.sentence_model.encode([proposal_text])[0]
        
        similarities = []
        for fid in team:
            if fid in self.faculty_profiles:
                sim = cosine_similarity([self.faculty_profiles[fid]['sentence_embeddings']], 
                                      [proposal_embedding])[0][0]
                similarities.append(sim)
        
        features.append(np.mean(similarities) if similarities else 0)
        features.append(np.std(similarities) if similarities else 0)
        
        # Collaboration history
        collab_count = 0
        for i, fid1 in enumerate(team):
            for fid2 in team[i+1:]:
                if self.collaboration_graph.has_edge(fid1, fid2):
                    collab_count += 1
        
        features.append(collab_count)
        
        # Publication overlap
        pub_years = []
        for fid in team:
            if fid in self.faculty_profiles:
                pub_years.extend(self.faculty_profiles[fid].get('publication_years', []))
        
        features.append(len(set(pub_years)) if pub_years else 0)
        
        # Method diversity
        all_methods = []
        for fid in team:
            if fid in self.faculty_profiles:
                all_methods.extend(self.faculty_profiles[fid].get('research_methods', []))
        
        features.append(len(set(all_methods)) if all_methods else 0)
        
        return features
    
    def _get_feature_names(self):
        """Get feature names for ML model"""
        return [
            'team_size',
            'department_diversity',
            'avg_similarity',
            'std_similarity',
            'collaboration_count',
            'publication_year_span',
            'method_diversity'
        ]
    
    def _optimize_weights_from_ml(self, feature_importances):
        """Optimize weights based on ML feature importances"""
        # Map features to weight categories
        importance_sum = sum(feature_importances)
        
        # Update weights based on feature importances
        self.optimal_weights['diversity_weight'] = feature_importances[1] / importance_sum
        self.optimal_weights['embedding_weight'] = feature_importances[2] / importance_sum
        self.optimal_weights['collaboration_weight'] = feature_importances[4] / importance_sum
        
        print("Optimized weights from ML model:", self.optimal_weights)
    
    def _train_synthetic_model(self):
        """Train a model with synthetic data when no historical data is available"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X = np.random.rand(n_samples, 7)
        
        # Generate labels with some logical relationship to features
        y = (0.3 * X[:, 1] +  # department diversity
             0.4 * X[:, 2] +  # avg similarity
             0.2 * X[:, 4] +  # collaboration count
             0.1 * X[:, 6] +  # method diversity
             0.1 * np.random.rand(n_samples))  # noise
        
        # Normalize to 0-1 range
        y = (y - y.min()) / (y.max() - y.min())
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train model
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ml_model.fit(X_scaled, y)
        
        self.feature_names = self._get_feature_names()
    
    def predict_collaborations_with_ml(self, grant_proposal_text, top_n=5, team_sizes=[2, 3], use_ml=True):
        """
        Enhanced prediction using ML model and knowledge graph
        
        Parameters:
        grant_proposal_text (str): The grant proposal text
        top_n (int): Number of recommendations
        team_sizes (list): Team sizes to consider
        use_ml (bool): Whether to use ML model for scoring
        
        Returns:
        list: Top recommended collaborations
        """
        # Analyze grant proposal
        proposal_analysis = self.analyze_grant_proposal_enhanced(grant_proposal_text)
        
        # Get faculty matches using multiple methods
        faculty_matches = self._match_faculty_enhanced(proposal_analysis)
        
        # Find optimal combinations
        if use_ml and self.ml_model:
            top_combinations = self._find_optimal_combinations_ml(
                faculty_matches, 
                proposal_analysis,
                top_n=top_n,
                team_sizes=team_sizes
            )
        else:
            top_combinations = self._find_optimal_combinations(
                faculty_matches,
                top_n=top_n,
                team_sizes=team_sizes,
                **self.optimal_weights
            )
        
        return top_combinations
    
    def analyze_grant_proposal_enhanced(self, proposal_text):
        """
        Enhanced grant proposal analysis with knowledge graph integration
        
        Parameters:
        proposal_text (str): Grant proposal text
        
        Returns:
        dict: Comprehensive analysis
        """
        # Process with spaCy
        doc = self.nlp(proposal_text)
        
        # Extract embeddings
        bert_embedding = self.extract_contextual_embeddings(proposal_text)
        sentence_embedding = self.sentence_model.encode([proposal_text])[0]
        
        # Extract keywords using multiple methods
        advanced_keywords = self.extract_advanced_keywords(proposal_text)
        
        # Extract entities with potential links
        entities = self._extract_entities_with_links(doc)
        
        # Topic modeling
        if self.dictionary and self.lda_model:
            tokenized = [token.lemma_.lower() for token in doc 
                        if not token.is_stop and not token.is_punct and token.is_alpha]
            bow = self.dictionary.doc2bow(tokenized)
            topic_distribution = self.lda_model[bow]
        else:
            topic_distribution = []
        
        # Extract specific grant requirements
        requirements = self._extract_grant_requirements(proposal_text)
        
        return {
            'text': proposal_text,
            'bert_embedding': bert_embedding,
            'sentence_embedding': sentence_embedding,
            'keywords': self._extract_keywords(doc),
            'advanced_keywords': advanced_keywords,
            'entities': entities,
            'topic_distribution': topic_distribution,
            'requirements': requirements,
            'research_methods': self._extract_research_methods(proposal_text),
            'interdisciplinary_keywords': self._extract_interdisciplinary_keywords(proposal_text)
        }
    
    def _extract_grant_requirements(self, text):
        """Extract specific requirements from grant text"""
        requirements = {
            'disciplines': [],
            'methods': [],
            'outcomes': [],
            'constraints': []
        }
        
        # Discipline requirements
        discipline_pattern = r'(?:require|need|must have).*?(?:from|in)\s+(\w+\s+(?:and\s+)?\w+)'
        for match in re.finditer(discipline_pattern, text, re.IGNORECASE):
            requirements['disciplines'].append(match.group(1))
        
        # Method requirements
        method_keywords = ['experimental', 'theoretical', 'computational', 'clinical']
        for keyword in method_keywords:
            if keyword in text.lower():
                requirements['methods'].append(keyword)
        
        return requirements
    
    def _extract_interdisciplinary_keywords(self, text):
        """Extract keywords indicating interdisciplinary nature"""
        interdisciplinary_indicators = [
            'interdisciplinary', 'cross-disciplinary', 'multidisciplinary',
            'collaborative', 'integrated', 'convergence', 'bridging'
        ]
        
        found_indicators = []
        text_lower = text.lower()
        
        for indicator in interdisciplinary_indicators:
            if indicator in text_lower:
                found_indicators.append(indicator)
        
        return found_indicators
    
    def _match_faculty_enhanced(self, proposal_analysis):
        """
        Enhanced faculty matching using knowledge graph and ML features
        
        Parameters:
        proposal_analysis (dict): Analysis results
        
        Returns:
        dict: Faculty matches with detailed scores
        """
        matches = {}
        
        # Query knowledge graph
        kg_scores = self.query_knowledge_graph(
            proposal_analysis['advanced_keywords'],
            proposal_analysis['entities']
        )
        
        for faculty_id, profile in self.faculty_profiles.items():
            # BERT embedding similarity
            bert_similarity = cosine_similarity(
                [profile['bert_embeddings']], 
                [proposal_analysis['bert_embedding']]
            )[0][0]
            
            # Sentence embedding similarity
            sentence_similarity = cosine_similarity(
                [profile['sentence_embeddings']], 
                [proposal_analysis['sentence_embedding']]
            )[0][0]
            
            # Advanced keyword matching
            faculty_keywords = dict(profile['advanced_keywords'])
            proposal_keywords = dict(proposal_analysis['advanced_keywords'])
            
            keyword_score = 0
            for keyword, score in proposal_keywords.items():
                if keyword in faculty_keywords:
                    keyword_score += score * faculty_keywords[keyword]
            
            # Normalize keyword score
            keyword_score = keyword_score / max(1, len(proposal_keywords))
            
            # Entity matching with semantic similarity
            entity_score = self._calculate_entity_score(
                profile.get('entities', {}),
                proposal_analysis.get('entities', {})
            )
            
            # Topic similarity
            topic_score = self._calculate_topic_similarity(
                profile, 
                proposal_analysis['topic_distribution']
            )
            
            # Knowledge graph score
            kg_score = kg_scores.get(faculty_id, 0)
            
            # Method matching
            method_score = self._calculate_method_match(
                profile.get('research_methods', []),
                proposal_analysis.get('research_methods', [])
            )
            
            # Calculate weighted overall score
            weights = self.optimal_weights
            overall_score = (
                weights['embedding_weight'] * (bert_similarity + sentence_similarity) / 2 +
                weights['keyword_weight'] * keyword_score +
                weights['entity_weight'] * entity_score +
                weights['topic_weight'] * topic_score +
                weights['knowledge_graph_weight'] * kg_score +
                0.1 * method_score  # Additional weight for method matching
            )
            
            # Store comprehensive match data
            matches[faculty_id] = {
                'faculty_name': profile['name'],
                'department': profile['department'],
                'bert_similarity': bert_similarity,
                'sentence_similarity': sentence_similarity,
                'keyword_score': keyword_score,
                'entity_score': entity_score,
                'topic_score': topic_score,
                'kg_score': kg_score,
                'method_score': method_score,
                'overall_score': overall_score,
                'match_explanation': self._generate_detailed_explanation(
                    bert_similarity, sentence_similarity, keyword_score, 
                    entity_score, topic_score, kg_score, method_score
                ),
                'expertise_evolution': self._analyze_expertise_evolution(profile),
                'collaboration_strength': self._calculate_collaboration_strength(faculty_id)
            }
        
        return matches
    
    def _calculate_entity_score(self, faculty_entities, proposal_entities):
        """Calculate entity matching score with semantic similarity"""
        if not faculty_entities or not proposal_entities:
            return 0.0
        
        score = 0
        total_weight = 0
        
        # Weight different entity types
        entity_weights = {
            'ORG': 1.0,
            'PERSON': 0.5,
            'LOC': 0.3,
            'TECH': 1.5,  # Custom entity type for technologies
            'METHOD': 1.2  # Custom entity type for methods
        }
        
        for entity_type, proposal_ents in proposal_entities.items():
            weight = entity_weights.get(entity_type, 0.8)
            total_weight += weight * len(proposal_ents)
            
            if entity_type in faculty_entities:
                faculty_ents = faculty_entities[entity_type]
                
                # Extract just the text if entities are dictionaries
                if isinstance(proposal_ents[0], dict):
                    proposal_texts = [e['text'] for e in proposal_ents]
                else:
                    proposal_texts = proposal_ents
                
                if isinstance(faculty_ents[0], dict):
                    faculty_texts = [e['text'] for e in faculty_ents]
                else:
                    faculty_texts = faculty_ents
                
                # Calculate matches
                matches = len(set(faculty_texts) & set(proposal_texts))
                score += weight * matches
        
        return score / max(1, total_weight)
    
    def _calculate_topic_similarity(self, profile, proposal_topics):
        """Calculate topic similarity using advanced methods"""
        if not proposal_topics or not self.lda_model:
            return 0.0
        
        # Get faculty topic distribution
        faculty_text = profile['bio_text']
        doc = self.nlp(faculty_text)
        tokenized = [token.lemma_.lower() for token in doc 
                    if not token.is_stop and not token.is_punct and token.is_alpha]
        
        if not tokenized:
            return 0.0
        
        bow = self.dictionary.doc2bow(tokenized)
        faculty_topics = self.lda_model[bow]
        
        # Calculate Jensen-Shannon divergence or cosine similarity
        faculty_topic_vec = np.zeros(self.num_topics)
        proposal_topic_vec = np.zeros(self.num_topics)
        
        for topic_id, prob in faculty_topics:
            faculty_topic_vec[topic_id] = prob
        
        for topic_id, prob in proposal_topics:
            proposal_topic_vec[topic_id] = prob
        
        # Use cosine similarity for topic vectors
        similarity = cosine_similarity([faculty_topic_vec], [proposal_topic_vec])[0][0]
        
        return similarity
    
    def _calculate_method_match(self, faculty_methods, proposal_methods):
        """Calculate score based on matching research methods."""
        if not faculty_methods or not proposal_methods:
            return 0.0
        
        matched_methods = len(set(faculty_methods) & set(proposal_methods))
        return matched_methods / max(len(proposal_methods), 1)

    def _generate_detailed_explanation(self, bert_sim, sent_sim, keyword_score, entity_score, topic_score, kg_score, method_score):
        """Generate detailed explanation for faculty match"""
        explanations = []

        # Semantic similarity
        avg_embedding_sim = (bert_sim + sent_sim) / 2
        if avg_embedding_sim > 0.8:
            explanations.append("Exceptional semantic alignment with grant objectives")
        elif avg_embedding_sim > 0.65:
            explanations.append("Strong semantic alignment with grant themes")
        elif avg_embedding_sim > 0.5:
            explanations.append("Good semantic alignment with proposal")

        # Keyword matching
        if keyword_score > 0.7:
            explanations.append("Extensive expertise overlap in key research areas")
        elif keyword_score > 0.5:
            explanations.append("Significant keyword alignment with grant requirements")
        elif keyword_score > 0.3:
            explanations.append("Moderate keyword overlap with proposal themes")

        # Entity matching
        if entity_score > 0.6:
            explanations.append("Strong familiarity with key entities and organizations")
        elif entity_score > 0.4:
            explanations.append("Experience with relevant entities mentioned")
        
        # Topic modeling
        if topic_score > 0.7:
            explanations.append("Research topics closely match grant focus areas")
        elif topic_score > 0.5:
            explanations.append("Good topical alignment with proposal")

        # Knowledge graph
        if kg_score > 0.8:
            explanations.append("Knowledge graph indicates exceptional domain expertise")
        elif kg_score > 0.6:
            explanations.append("Strong knowledge graph connections to grant topics")
        
        # Method matching
        if method_score > 0.7:
            explanations.append("Expertise in required research methodologies")
        elif method_score > 0.5:
            explanations.append("Experience with some required methods")

        return explanations

    def _analyze_expertise_evolution(self, profile):
        """Analyze how faculty expertise has evolved over time"""
        pub_years = profile.get('publication_years', [])
        if not pub_years:
            return {'trend': 'unknown', 'recent_activity': False}
        
        current_year = datetime.now().year
        recent_years = [y for y in pub_years if y >= current_year - 5]
        
        return {
            'trend': 'active' if len(recent_years) > 0 else 'dormant',
            'recent_activity': len(recent_years) > 0
        }

    def _calculate_collaboration_strength(self, faculty_id):
        """Calculate a faculty's collaboration strength based on graph centrality"""
        if faculty_id not in self.collaboration_graph:
            return 0
        
        # Use degree centrality as a simple measure of collaboration strength
        # Higher degree means more direct collaborations
        return self.collaboration_graph.degree(faculty_id)
    
    def _find_optimal_combinations_ml(self, faculty_matches, proposal_analysis, top_n=5, team_sizes=[2, 3]):
        """
        Find optimal faculty combinations using ML model for scoring
        
        Parameters:
        faculty_matches (dict): Faculty matches with detailed scores
        proposal_analysis (dict): Analyzed grant proposal
        top_n (int): Number of top combinations to return
        team_sizes (list): List of desired team sizes to consider
        
        Returns:
        list: Top combinations with ML-based scoring
        """
        # Sort faculty by match score
        sorted_faculty = sorted(faculty_matches.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        # Limit pool size for computational efficiency
        max_team_size = max(team_sizes)
        pool_size = min(len(sorted_faculty), max(15, 3 * max_team_size))
        top_faculty = sorted_faculty[:pool_size]
        
        combinations = []
        from itertools import combinations as itertools_combinations
        
        for size in team_sizes:
            if size > len(top_faculty):
                continue
            for combo_indices in itertools_combinations(range(len(top_faculty)), size):
                combo = [top_faculty[i] for i in combo_indices]
                faculty_ids = [f[0] for f in combo]
                
                # Extract features for ML model
                features = self._extract_team_features(faculty_ids, proposal_analysis['text'])
                features_scaled = self.feature_scaler.transform([features])
                
                # Predict score using ML model
                ml_score = self.ml_model.predict(features_scaled)[0]
                
                # Calculate additional scores
                score_info = self._calculate_combination_score_enhanced(combo, proposal_analysis)
                
                # Combine ML score with other factors
                combined_score = 0.7 * ml_score + 0.3 * score_info['rule_based_score'] # Weighted average
                
                combinations.append({
                    'faculty': faculty_ids,
                    'faculty_names': [f[1]['faculty_name'] for f in combo],
                    'departments': [f[1]['department'] for f in combo],
                    'individual_scores': [f[1]['overall_score'] for f in combo],
                    'individual_explanations': [f[1]['match_explanation'] for f in combo],
                    'ml_score': ml_score,
                    'combined_score': combined_score,
                    'base_score': score_info['base_score'],
                    'collaboration_bonus': score_info['collaboration_bonus'],
                    'diversity_score': score_info['diversity_score'],
                    'complementarity_score': score_info['complementarity_score'],
                    'synergy_reasons': score_info['synergy_reasons'],
                    'collaboration_network': score_info['collaboration_network'],
                    'expertise_coverage': score_info['expertise_coverage'],
                    'team_size': size
                })
        
        # Sort combinations by combined score
        sorted_combinations = sorted(combinations, key=lambda x: x['combined_score'], reverse=True)
        return sorted_combinations[:top_n]

    def _find_optimal_combinations(self, faculty_matches, top_n=5, team_sizes=[2, 3], **kwargs):
        """
        Find optimal faculty combinations based on weighted scores (rule-based)
        This is a fallback or alternative to the ML-based approach.
        """
        # Sort faculty by match score
        sorted_faculty = sorted(faculty_matches.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        # Limit pool size for computational efficiency
        max_team_size = max(team_sizes)
        faculty_pool_size = min(len(sorted_faculty), max(10, 3 * max_team_size))
        top_faculty = sorted_faculty[:faculty_pool_size]
        
        combinations = []
        from itertools import combinations as itertools_combinations
        
        for size in team_sizes:
            if size > len(top_faculty):
                continue
            for combo_indices in itertools_combinations(range(len(top_faculty)), size):
                combo = [top_faculty[i] for i in combo_indices]
                
                # Calculate combined score
                score_info = self._calculate_combination_score(
                    combo,
                    collaboration_weight=kwargs.get('collaboration_weight', 0.1),
                    diversity_weight=kwargs.get('diversity_weight', 0.2)
                )
                
                combinations.append({
                    'faculty': [f[0] for f in combo],
                    'faculty_names': [f[1]['faculty_name'] for f in combo],
                    'departments': [f[1]['department'] for f in combo],
                    'individual_scores': [f[1]['overall_score'] for f in combo],
                    'individual_explanations': [f[1].get('match_explanation', []) for f in combo],
                    'combined_score': score_info['combined_score'],
                    'base_score': score_info['base_score'],
                    'collaboration_bonus': score_info['collaboration_bonus'],
                    'diversity_score': score_info['diversity_score'],
                    'complementarity_score': score_info['complementarity_score'],
                    'synergy_reasons': score_info['synergy_reasons'],
                    'collaboration_details': score_info['collaboration_details'],
                    'expertise_areas': score_info['expertise_areas'],
                    'team_size': size
                })
        
        # Sort combinations by combined score
        sorted_combinations = sorted(combinations, key=lambda x: x['combined_score'], reverse=True)
        return sorted_combinations[:top_n]

    def _calculate_combination_score_enhanced(self, faculty_combo, proposal_analysis):
        """
        Calculate a comprehensive score for a faculty combination,
        considering expertise alignment, collaboration, diversity, and complementarity.
        """
        if not faculty_combo:
            return {
                'rule_based_score': 0.0, 'base_score': 0.0, 'collaboration_bonus': 0.0,
                'diversity_score': 0.0, 'complementarity_score': 0.0,
                'synergy_reasons': [], 'collaboration_network': {}, 'expertise_coverage': {}
            }

        faculty_ids = [f[0] for f in faculty_combo]
        
        # 1. Base expertise match (average of individual overall scores)
        individual_overall_scores = [f[1]['overall_score'] for f in faculty_combo]
        base_score = np.mean(individual_overall_scores)

        # 2. Collaboration history bonus
        collaboration_network_analysis = self._analyze_collaboration_network(faculty_ids)
        collaboration_bonus = collaboration_network_analysis['collaboration_score'] * self.optimal_weights['collaboration_weight']

        # 3. Team diversity bonus
        diversity_analysis = self._analyze_team_diversity(faculty_ids)
        # Assuming we want to maximize diversity for certain metrics, e.g., department, methods, topics
        diversity_score = (diversity_analysis['department_diversity'] * 0.4 +
                           diversity_analysis['method_diversity'] * 0.3 +
                           diversity_analysis['topic_diversity'] * 0.3) * self.optimal_weights['diversity_weight']

        # 4. Complementarity score
        complementarity_analysis = self._analyze_team_complementarity(faculty_ids)
        complementarity_score = complementarity_analysis['score'] * self.optimal_weights['diversity_weight'] # Re-using diversity weight for now

        # 5. Expertise coverage (how well the team covers grant keywords)
        expertise_coverage_analysis = self._analyze_expertise_coverage(faculty_ids, proposal_analysis['advanced_keywords'])
        expertise_coverage_bonus = expertise_coverage_analysis['coverage_score'] * 0.1 # Small bonus for coverage

        # Combined rule-based score
        rule_based_score = base_score + collaboration_bonus + diversity_score + complementarity_score + expertise_coverage_bonus

        # Generate synergy reasons
        synergy_reasons = self._generate_team_synergy_reasons(
            base_score, collaboration_network_analysis, diversity_analysis,
            complementarity_analysis, expertise_coverage_analysis
        )
        
        return {
            'rule_based_score': rule_based_score,
            'base_score': base_score,
            'collaboration_bonus': collaboration_bonus,
            'diversity_score': diversity_score,
            'complementarity_score': complementarity_score,
            'synergy_reasons': synergy_reasons,
            'collaboration_network': collaboration_network_analysis,
            'expertise_coverage': expertise_coverage_analysis
        }

    def _calculate_combination_score(self, faculty_combo, collaboration_weight=0.1, diversity_weight=0.2):
        """Original combination scoring method for backward compatibility"""
        # Base score is average of individual scores
        individual_scores = [f[1]['overall_score'] for f in faculty_combo]
        base_score = sum(individual_scores) / len(individual_scores)

        # Collaboration bonus
        collaboration_bonus = 0
        collaboration_details = []
        unique_departments = set()
        expertise_areas = set()

        # Gather collaboration and diversity info
        for i, (fid1, profile1) in enumerate(faculty_combo):
            unique_departments.add(profile1['department'])
            for kw, _ in profile1.get('advanced_keywords', []):
                expertise_areas.add(kw)
            
            for j, (fid2, profile2) in enumerate(faculty_combo):
                if i < j and self.collaboration_graph.has_edge(fid1, fid2):
                    weight = self.collaboration_graph[fid1][fid2].get('weight', 1)
                    collaboration_bonus += weight * collaboration_weight
                    collaboration_details.append(f"{profile1['name']} and {profile2['name']} have collaborated (strength: {weight})")

        # Diversity score (e.g., department diversity)
        expertise_diversity_score = (len(unique_departments) / len(faculty_combo)) * diversity_weight if faculty_combo else 0
        
        # Complementarity score (simple version: penalize high keyword overlap)
        complementarity_score = 0
        all_keywords = []
        for _, profile in faculty_combo:
            all_keywords.extend([kw for kw, _ in profile.get('advanced_keywords', [])])
        
        if all_keywords:
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            # Higher score for less common keywords within the group (more unique contribution)
            complementarity_score = sum(1/count for count in keyword_counts.values()) / len(all_keywords)
        
        # Combined score formula
        combined_score = base_score + collaboration_bonus + expertise_diversity_score + complementarity_score
        
        # Generate explanation for the combination
        synergy_reasons = []
        if len(unique_departments) > 1:
            synergy_reasons.append(f"Interdisciplinary team spanning {len(unique_departments)} departments")
        if collaboration_bonus > 0:
            collab_count = len(collaboration_details)
            if collab_count == 1:
                synergy_reasons.append("These faculty have worked together previously")
            else:
                synergy_reasons.append(f"Strong collaboration network with {collab_count} prior collaborations")
        if len(expertise_areas) > 10:
            synergy_reasons.append("Wide range of combined expertise relevant to the proposal")
        if complementarity_score > 0.05:
            synergy_reasons.append("Faculty have complementary rather than overlapping expertise")

        # Return comprehensive information
        return {
            'combined_score': combined_score,
            'base_score': base_score,
            'collaboration_bonus': collaboration_bonus,
            'diversity_score': expertise_diversity_score,
            'complementarity_score': complementarity_score,
            'synergy_reasons': synergy_reasons,
            'collaboration_details': collaboration_details,
            'unique_departments': len(unique_departments),
            'expertise_areas': list(expertise_areas)
        }

    def _analyze_collaboration_network(self, faculty_ids):
        """Analyze the collaboration network within a given team."""
        network_stats = {
            'total_collaborations': 0,
            'collaboration_pairs': [],
            'network_density': 0,
            'average_collaboration_strength': 0,
            'collaboration_score': 0
        }
        
        if len(faculty_ids) < 2:
            return network_stats

        collaboration_weights = []
        for i, fid1 in enumerate(faculty_ids):
            for j, fid2 in enumerate(faculty_ids[i+1:], i+1):
                if self.collaboration_graph.has_edge(fid1, fid2):
                    weight = self.collaboration_graph[fid1][fid2].get('weight', 1)
                    network_stats['total_collaborations'] += 1
                    collaboration_weights.append(weight)
                    network_stats['collaboration_pairs'].append({
                        'faculty1': self.faculty_profiles[fid1]['name'],
                        'faculty2': self.faculty_profiles[fid2]['name'],
                        'strength': weight
                    })
        
        # Calculate network density
        max_possible_edges = len(faculty_ids) * (len(faculty_ids) - 1) / 2
        if max_possible_edges > 0:
            network_stats['network_density'] = network_stats['total_collaborations'] / max_possible_edges

        # Average collaboration strength
        if collaboration_weights:
            network_stats['average_collaboration_strength'] = np.mean(collaboration_weights)
        
        # Overall collaboration score (0-1)
        # Normalize average strength by a max expected weight (e.g., 5, adjust as needed)
        network_stats['collaboration_score'] = (
            0.6 * network_stats['network_density'] + 
            0.4 * min(network_stats['average_collaboration_strength'] / 5, 1) 
        )
        
        return network_stats

    def _analyze_team_diversity(self, faculty_ids):
        """Comprehensive diversity analysis of a team"""
        diversity_metrics = {
            'department_diversity': 0,
            'method_diversity': 0,
            'topic_diversity': 0,
            'career_stage_diversity': 0, # Placeholder, needs actual data
            'overall_diversity': 0
        }

        if not faculty_ids:
            return diversity_metrics

        # Department diversity
        departments = [self.faculty_profiles[fid]['department'] for fid in faculty_ids if fid in self.faculty_profiles]
        unique_departments = len(set(departments))
        diversity_metrics['department_diversity'] = unique_departments / len(departments) if departments else 0

        # Method diversity
        all_methods = []
        for fid in faculty_ids:
            if fid in self.faculty_profiles:
                all_methods.extend(self.faculty_profiles[fid].get('research_methods', []))
        unique_methods = len(set(all_methods))
        diversity_metrics['method_diversity'] = unique_methods / max(len(all_methods), 1)

        # Topic diversity (using LDA topic distributions)
        if self.lda_model and self.dictionary:
            team_topic_vectors = []
            for fid in faculty_ids:
                if fid in self.faculty_profiles:
                    profile = self.faculty_profiles[fid]
                    doc = self.nlp(profile['bio_text'])
                    tokenized = [token.lemma_.lower() for token in doc 
                                if not token.is_stop and not token.is_punct and token.is_alpha]
                    
                    if tokenized:
                        bow = self.dictionary.doc2bow(tokenized)
                        topic_distribution = self.lda_model[bow]
                        vec = np.zeros(self.num_topics)
                        for topic_id, prob in topic_distribution:
                            vec[topic_id] = prob
                        team_topic_vectors.append(vec)
            
            if len(team_topic_vectors) > 1:
                # Calculate average pairwise cosine distance (1 - similarity)
                pairwise_distances = []
                for i in range(len(team_topic_vectors)):
                    for j in range(i + 1, len(team_topic_vectors)):
                        sim = cosine_similarity([team_topic_vectors[i]], [team_topic_vectors[j]])[0][0]
                        pairwise_distances.append(1 - sim)
                diversity_metrics['topic_diversity'] = np.mean(pairwise_distances)
            else:
                diversity_metrics['topic_diversity'] = 0.0 # Only one or no members for topic diversity

        # Overall diversity (simple average of calculated diversities)
        diversity_scores_calculated = [
            diversity_metrics['department_diversity'],
            diversity_metrics['method_diversity'],
            diversity_metrics['topic_diversity']
        ]
        diversity_metrics['overall_diversity'] = np.mean(diversity_scores_calculated)

        return diversity_metrics

    def _analyze_expertise_coverage(self, faculty_ids, grant_keywords):
        """Analyze how well the team's combined expertise covers the grant's keywords."""
        coverage = {
            'covered_keywords': [],
            'missing_keywords': [],
            'keyword_coverage': 0.0,
            'coverage_score': 0.0
        }

        if not faculty_ids or not grant_keywords:
            return coverage

        # Collect all faculty keywords
        faculty_keywords = set()
        for fid in faculty_ids:
            if fid in self.faculty_profiles:
                keywords = [kw for kw, _ in self.faculty_profiles[fid].get('advanced_keywords', [])[:15]]
                faculty_keywords.update(keywords)
        
        # Check coverage of grant keywords
        grant_keyword_set = set([kw for kw, _ in grant_keywords])
        covered = faculty_keywords & grant_keyword_set
        missing = grant_keyword_set - faculty_keywords

        coverage['covered_keywords'] = list(covered)
        coverage['missing_keywords'] = list(missing)
        
        if grant_keyword_set:
            coverage['keyword_coverage'] = len(covered) / len(grant_keyword_set)
        else:
            coverage['keyword_coverage'] = 0.0 # No grant keywords to cover

        coverage['coverage_score'] = coverage['keyword_coverage'] # Simple score for now
        return coverage

    def _analyze_team_complementarity(self, faculty_ids):
        """Analyze how complementary team members are in terms of unique expertise."""
        complementarity = {
            'keyword_overlap': 0,
            'unique_contributions': {},
            'redundancy': 0,
            'score': 0
        }

        if len(faculty_ids) < 2:
            complementarity['score'] = 0.5 # Neutral score for single member or empty
            return complementarity

        # Analyze keyword overlap
        keyword_sets = {}
        all_keywords = set()
        for fid in faculty_ids:
            if fid in self.faculty_profiles:
                keywords = set([kw for kw, _ in self.faculty_profiles[fid].get('advanced_keywords', [])[:20]])
                keyword_sets[fid] = keywords
                all_keywords.update(keywords)

        # Calculate unique contributions for each faculty member
        for fid, keywords in keyword_sets.items():
            unique = keywords.copy()
            for other_fid, other_keywords in keyword_sets.items():
                if fid != other_fid:
                    unique -= other_keywords
            name = self.faculty_profiles[fid]['name']
            complementarity['unique_contributions'][name] = len(unique)
        
        # Calculate redundancy (1 - (unique_keywords / total_keywords_sum))
        total_keywords = sum(len(kw_set) for kw_set in keyword_sets.values())
        unique_keywords = len(all_keywords)
        if total_keywords > 0:
            complementarity['redundancy'] = 1 - (unique_keywords / total_keywords)

        # Keyword overlap (average pairwise Jaccard similarity)
        pairwise_jaccard = []
        for i in range(len(faculty_ids)):
            for j in range(i + 1, len(faculty_ids)):
                id1 = faculty_ids[i]
                id2 = faculty_ids[j]
                if id1 in keyword_sets and id2 in keyword_sets:
                    set1 = keyword_sets[id1]
                    set2 = keyword_sets[id2]
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    if union > 0:
                        pairwise_jaccard.append(intersection / union)
        
        if pairwise_jaccard:
            complementarity['keyword_overlap'] = np.mean(pairwise_jaccard)
        
        # Complementarity score: Penalize high overlap, reward unique contributions
        # A simple heuristic: (1 - average_overlap) + (average_unique_contributions_normalized)
        avg_overlap_penalty = complementarity['keyword_overlap']
        avg_unique_contrib_score = np.mean(list(complementarity['unique_contributions'].values())) / 20 # Max 20 keywords
        
        complementarity['score'] = (1 - avg_overlap_penalty) * 0.7 + avg_unique_contrib_score * 0.3 # Weighted combination
        
        return complementarity

    def _generate_team_synergy_reasons(self, base_score, collab_analysis, diversity_analysis, complementarity_analysis, expertise_coverage_analysis):
        """Generate detailed reasons for team synergy."""
        reasons = []

        if base_score > 0.7:
            reasons.append("High overall expertise alignment with the grant proposal.")
        elif base_score > 0.5:
            reasons.append("Good individual expertise match with the grant proposal.")
        
        if collab_analysis['total_collaborations'] > 0:
            if collab_analysis['network_density'] > 0.5:
                reasons.append("Strong existing collaboration network within the team, indicating proven teamwork.")
            elif collab_analysis['total_collaborations'] > 0:
                reasons.append(f"Team members have {collab_analysis['total_collaborations']} prior collaborations, suggesting established working relationships.")
        
        if diversity_analysis['department_diversity'] > 0.5:
            reasons.append("Excellent departmental diversity, fostering interdisciplinary perspectives.")
        if diversity_analysis['method_diversity'] > 0.5:
            reasons.append("Diverse methodological expertise, enabling a multi-faceted research approach.")
        if diversity_analysis['topic_diversity'] > 0.5:
            reasons.append("Broad coverage of research topics, enhancing comprehensive project understanding.")

        if complementarity_analysis['score'] > 0.7:
            reasons.append("Highly complementary team members, ensuring broad and non-redundant expertise coverage.")
        elif complementarity_analysis['score'] > 0.5:
            reasons.append("Good complementarity among team members, suggesting unique contributions.")
        
        if expertise_coverage_analysis['keyword_coverage'] > 0.8:
            reasons.append("Comprehensive coverage of key grant keywords by the team's combined expertise.")
        elif expertise_coverage_analysis['keyword_coverage'] > 0.6:
            reasons.append("Strong coverage of essential grant keywords.")
        
        return reasons


    def _analyze_modification_impact(self, original_details, modified_details):
        """Analyze the impact of a team modification on key metrics."""
        impact = {}
        
        # Compare key scores
        impact['combined_score_change'] = modified_details['rule_based_score'] - original_details['rule_based_score']
        impact['base_score_change'] = modified_details['base_score'] - original_details['base_score']
        impact['collaboration_bonus_change'] = modified_details['collaboration_bonus'] - original_details['collaboration_bonus']
        impact['diversity_score_change'] = modified_details['diversity_score'] - original_details['diversity_score']
        impact['complementarity_score_change'] = modified_details['complementarity_score'] - original_details['complementarity_score']

        # Determine overall assessment
        if impact['combined_score_change'] > 0.1:
            impact['assessment'] = "Significant Improvement"
        elif impact['combined_score_change'] > 0.02:
            impact['assessment'] = "Moderate Improvement"
        elif impact['combined_score_change'] < -0.1:
            impact['assessment'] = "Significant Deterioration"
        elif impact['combined_score_change'] < -0.02:
            impact['assessment'] = "Moderate Deterioration"
        else:
            impact['assessment'] = "Minor Change"
        
        # Provide specific insights
        insights = []
        if impact['base_score_change'] > 0.05:
            insights.append("Improved individual expertise match with the grant.")
        if impact['collaboration_bonus_change'] > 0.02:
            insights.append("Enhanced team collaboration network.")
        if impact['diversity_score_change'] > 0.05:
            insights.append("Increased team diversity (e.g., departments, methods).")
        if impact['complementarity_score_change'] > 0.05:
            insights.append("Greater complementarity and unique contributions.")
        if impact['combined_score_change'] < 0:
            # If overall score decreased, explain why it might be a negative change
            if abs(impact['base_score_change']) > 0.05 and impact['base_score_change'] < 0:
                insights.append("Reduced individual expertise match.")
            if abs(impact['collaboration_bonus_change']) > 0.02 and impact['collaboration_bonus_change'] < 0:
                insights.append("Weakened team collaboration network.")
            if abs(impact['diversity_score_change']) > 0.05 and impact['diversity_score_change'] < 0:
                insights.append("Decreased team diversity.")

        impact['insights'] = insights if insights else ["No significant specific impact observed."]
        
        return impact

    def suggest_team_modifications(self, current_team_ids, grant_proposal_text, top_n=3):
        """
        Suggest modifications (additions or replacements) to an existing team.
        """
        if not current_team_ids:
            return {"scenarios": [], "assessment": "No current team provided."}

        # Analyze current team
        current_team_profiles = [(fid, self.faculty_profiles[fid]) for fid in current_team_ids if fid in self.faculty_profiles]
        proposal_analysis = self.analyze_grant_proposal_enhanced(grant_proposal_text)
        
        original_details = self._calculate_combination_score_enhanced(current_team_profiles, proposal_analysis)
        original_ml_score = 0.5 # Default if no ML model
        if self.ml_model:
            original_features = self._extract_team_features(current_team_ids, grant_proposal_text)
            original_features_scaled = self.feature_scaler.transform([original_features])
            original_ml_score = self.ml_model.predict(original_features_scaled)[0]

        results = {
            'base_team': current_team_ids,
            'base_team_names': [self.faculty_profiles[fid]['name'] for fid in current_team_ids if fid in self.faculty_profiles],
            'base_team_score': {
                'combined_score': 0.7 * original_ml_score + 0.3 * original_details['rule_based_score'],
                'ml_score': original_ml_score,
                'rule_based_score': original_details['rule_based_score'],
                'details': original_details
            },
            'scenarios': []
        }

        # Identify potential candidates (faculty not in the current team)
        all_faculty_ids = set(self.faculty_profiles.keys())
        current_team_set = set(current_team_ids)
        candidate_pool = list(all_faculty_ids - current_team_set)

        # Prioritize candidates based on individual match to proposal
        candidate_matches = self._match_faculty_enhanced(proposal_analysis)
        sorted_candidate_pool = sorted(
            [ (fid, candidate_matches[fid]['overall_score']) for fid in candidate_pool if fid in candidate_matches ],
            key=lambda x: x[1], reverse=True
        )

        # Consider additions
        print(f"DEBUG: Considering additions to current team: {results['base_team_names']}")
        for candidate_id, _ in sorted_candidate_pool[:top_n * 2]: # Look at more candidates for additions
            if candidate_id not in self.faculty_profiles:
                continue
            modified_team = current_team_ids + [candidate_id]
            scenario_type = f"Add {self.faculty_profiles[candidate_id]['name']}"
            
            mod_features = self._extract_team_features(modified_team, grant_proposal_text)
            mod_features_scaled = self.feature_scaler.transform([mod_features])
            mod_ml_score = self.ml_model.predict(mod_features_scaled)[0] if self.ml_model else 0.5
            
            mod_combo = [(fid, self.faculty_profiles[fid]) for fid in modified_team if fid in self.faculty_profiles]
            mod_detailed = self._calculate_combination_score_enhanced(mod_combo, proposal_analysis)
            mod_combined_score = 0.7 * mod_ml_score + 0.3 * mod_detailed['rule_based_score']

            scenario_result = {
                'type': scenario_type,
                'modified_team': modified_team,
                'ml_score': mod_ml_score,
                'combined_score': mod_combined_score,
                'score_change': mod_combined_score - results['base_team_score']['combined_score'],
                'details': mod_detailed,
                'impact_analysis': self._analyze_modification_impact(original_details, mod_detailed)
            }
            results['scenarios'].append(scenario_result)

        # Consider replacements (if team size is fixed or for optimization)
        print("DEBUG: Considering replacements in current team.")
        if len(current_team_ids) > 1: # Only replace if there's more than one member
            # Sort current team members by their individual match to the proposal (lowest match first for replacement)
            current_team_sorted_by_match = sorted(
                [(fid, candidate_matches.get(fid, {}).get('overall_score', 0)) for fid in current_team_ids],
                key=lambda x: x[1]
            )

            for i, (replaced_id, _) in enumerate(current_team_sorted_by_match[:min(top_n, len(current_team_ids))]):
                for candidate_id, _ in sorted_candidate_pool[:top_n * 2]: # Look at more candidates for replacement
                    if candidate_id not in self.faculty_profiles:
                        continue
                    if candidate_id == replaced_id: # Don't replace with self
                        continue
                    
                    modified_team = [fid for fid in current_team_ids if fid != replaced_id] + [candidate_id]
                    scenario_type = f"Replace {self.faculty_profiles[replaced_id]['name']} with {self.faculty_profiles[candidate_id]['name']}"

                    mod_features = self._extract_team_features(modified_team, grant_proposal_text)
                    mod_features_scaled = self.feature_scaler.transform([mod_features])
                    mod_ml_score = self.ml_model.predict(mod_features_scaled)[0] if self.ml_model else 0.5
                    
                    mod_combo = [(fid, self.faculty_profiles[fid]) for fid in modified_team if fid in self.faculty_profiles]
                    mod_detailed = self._calculate_combination_score_enhanced(mod_combo, proposal_analysis)
                    mod_combined_score = 0.7 * mod_ml_score + 0.3 * mod_detailed['rule_based_score']

                    scenario_result = {
                        'type': scenario_type,
                        'modified_team': modified_team,
                        'ml_score': mod_ml_score,
                        'combined_score': mod_combined_score,
                        'score_change': mod_combined_score - results['base_team_score']['combined_score'],
                        'details': mod_detailed,
                        'impact_analysis': self._analyze_modification_impact(original_details, mod_detailed)
                    }
                    results['scenarios'].append(scenario_result)

        # Sort all scenarios by combined_score_change in descending order
        results['scenarios'] = sorted(results['scenarios'], key=lambda x: x['score_change'], reverse=True)[:top_n]
        
        return results

    def save_model(self, model_path="collaboration_model.pkl"):
        """
        Save the trained ML model, scaler, and LDA model to a file.
        """
        model_data = {
            'ml_model': self.ml_model,
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'optimal_weights': self.optimal_weights,
            'lda_model': self.lda_model,
            'dictionary': self.dictionary
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_path="collaboration_model.pkl"):
        """
        Load a previously trained model
        
        Parameters:
        model_path (str): Path to the saved model
        
        Returns:
        bool: Success status
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.ml_model = model_data['ml_model']
            self.feature_scaler = model_data['feature_scaler']
            self.feature_names = model_data['feature_names']
            self.optimal_weights = model_data['optimal_weights']
            self.lda_model = model_data.get('lda_model')
            self.dictionary = model_data.get('dictionary')
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # Additional helper methods for the original class functions
    def _extract_name(self, text):
        """Extract faculty name from bio text"""
        doc = self.nlp(text[:500]) # Check beginning of text
        # Look for PERSON entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Validate it's likely a faculty name (not a reference)
                if ent.start_char < 200: # Usually at the beginning
                    return ent.text
        
        # Fallback patterns
        name_patterns = [
            r"(?:Dr\.|Professor|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1)
        return "Unknown Faculty"

    def _extract_department(self, text):
        """Extract department from bio text"""
        department_keywords = [
            r"Department of ([^\n,]+)",
            r"Dept\. of ([^\n,]+)",
            r"School of ([^\n,]+)",
            r"Division of ([^\n,]+)"
        ]
        for pattern in department_keywords:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to general academic unit
        academic_unit_patterns = [
            r"College of ([^\n,]+)",
            r"Faculty of ([^\n,]+)"
        ]
        for pattern in academic_unit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip() + " College/Faculty"
        
        return "Unknown Department"

    def _extract_research_interests(self, text):
        """Extract general research interests snippet"""
        match = re.search(r"research interests?:?\s*(.+?)(?:\.|\n|$)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No specific interests mentioned."

    def _extract_keywords(self, doc):
        """Extract keywords using spaCy tokenization and noun chunks"""
        keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and not token.is_punct:
                keywords.append(token.lemma_.lower())
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1 and not chunk.text.lower() in [k.lower() for k in keywords]:
                keywords.append(chunk.text.lower())
        
        # Add some domain-specific terms (words with specific patterns)
        tech_pattern = re.compile(r'\b\w+(?:ology|ics|tion|ment|ing)\b', re.IGNORECASE)
        for match in tech_pattern.finditer(doc.text):
            keywords.append(match.group().lower())
        
        return list(set(keywords)) # Remove duplicates

    def _extract_paper_title(self, text):
        """Extract paper title from text"""
        lines = text.split("\n")
        # Usually the title is one of the first non-empty lines
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 20 and len(line) < 200: # Check if it looks like a title (capitalized, no weird characters)
                if not line.startswith("Abstract") and not line.startswith("Introduction"):
                    return line
        return "Unknown Title"

    def _extract_paper_authors(self, text):
        """Extract paper authors from text"""
        authors = []
        # Look for author section
        author_section = None
        author_patterns = [
            r"Authors?:?\s*([^\n]+)",
            r"by\s+([^\n]+?)(?:\n|Abstract|Introduction)",
        ]
        for pattern in author_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                author_section = match.group(1)
                break
        
        if author_section:
            # Parse author names
            # Split by commas, semicolons, or "and"
            parts = re.split(r'[,;]|\band\b', author_section)
            for part in parts:
                part = part.strip()
                # Check if it looks like a name
                if re.match(r'^[A-Z][a-z]+\s+[A-Z]', part):
                    authors.append(part)
        
        # Also use NER as fallback if not authors:
        if not authors:
            doc = self.nlp(text[:1000])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    authors.append(ent.text)
        
        return authors[:15] # Limit number of authors

    def _extract_paper_abstract(self, text):
        """Extract paper abstract from text"""
        match = re.search(r"Abstract\s*[\n\r]+(.+?)(?:\n\r+\d+\.?\s*Introduction|\n\r+\d+\.?\s*I\s*\.\s*Introduction|\n\r+1\s*\.\s*Introduction|\n\r+\s*Keywords:|\n\r+\s*1\s*(\.|\s*)\s*Introduction|$)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No abstract found."

    def _extract_earliest_publication_year(self, text):
        """Extract the earliest publication year mentioned in the text."""
        years = self._extract_publication_years(text)
        if years:
            return min(years)
        return None

    def _extract_latest_publication_year(self, text):
        """Extract the latest publication year mentioned in the text."""
        years = self._extract_publication_years(text)
        if years:
            return max(years)
        return None

    def _build_topic_model(self, texts):
        """
        Builds an LDA topic model from a collection of texts.
        
        Parameters:
        texts (list): A list of strings, where each string is a document.
        """
        if not texts:
            print("DEBUG: _build_topic_model received an empty list of texts. Skipping LDA.")
            self.lda_model = None
            self.lda_dictionary = None
            return

        print(f"DEBUG: _build_topic_model starting with {len(texts)} documents.")

        processed_texts = []
        for i, doc_text in enumerate(texts):
            if not isinstance(doc_text, str) or not doc_text.strip():
                print(f"DEBUG: Document {i} is empty or not a string. Skipping.")
                continue

            doc = self.nlp(doc_text.lower())
            tokens = [
                token.lemma_ for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha and len(token.lemma_) > 2
            ]
            if not tokens:
                print(f"DEBUG: Document {i} yielded no tokens after preprocessing. Original length: {len(doc_text)}")
            processed_texts.append(tokens)

        # Filter out empty documents from processed_texts
        processed_texts = [doc for doc in processed_texts if doc]
        print(f"DEBUG: Number of documents after preprocessing (non-empty): {len(processed_texts)}")

        if not processed_texts:
            print("ERROR: All documents became empty after preprocessing. Cannot build LDA model.")
            self.lda_model = None
            self.lda_dictionary = None
            return # Exit early if no processed texts

        self.lda_dictionary = corpora.Dictionary(processed_texts)
        print(f"DEBUG: Dictionary built with {len(self.lda_dictionary)} unique tokens.")
        
        if not self.lda_dictionary: # Check if dictionary is empty
            print("ERROR: Dictionary is empty. This means no terms were found across all processed documents.")
            self.lda_model = None
            self.lda_dictionary = None
            return

        corpus = [self.lda_dictionary.doc2bow(text) for text in processed_texts]
        print(f"DEBUG: Corpus built with {len(corpus)} documents.")

        # Filter out empty vectors from the corpus (documents that became empty after doc2bow)
        corpus = [vector for vector in corpus if vector]
        print(f"DEBUG: Corpus after filtering empty vectors: {len(corpus)}")

        if not corpus: # This is the final check before LDA model creation
            print("ERROR: Corpus is empty after converting to bag-of-words. Cannot compute LDA.")
            self.lda_model = None
            self.lda_dictionary = None
            return

        print(f"DEBUG: Attempting to build LDA model with {len(corpus)} documents and {len(self.lda_dictionary)} terms.")
        # Ensure num_topics is appropriate and consistent
        # num_topics must be less than or equal to the number of documents in the corpus
        num_topics = min(self.num_topics, len(corpus)) 
        if num_topics == 0: # If corpus is too small, cannot build any topics
            print("ERROR: Cannot set num_topics to 0 as corpus is too small. Exiting LDA model build.")
            self.lda_model = None
            self.lda_dictionary = None
            return


        self.lda_model = LdaModel(corpus,
                                  num_topics=num_topics,
                                  id2word=self.lda_dictionary,
                                  passes=10,
                                  random_state=100)
        print("DEBUG: LDA Model built successfully.")

    def load_research_papers(self, directory_path):
        """Load and process research papers with enhanced features"""
        print(f"Loading research papers from {directory_path}")
        paper_texts = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt') or filename.endswith('.pdf'):
                paper_id = os.path.splitext(filename)[0]
                file_path = os.path.join(directory_path, filename)
                
                content = ""
                try:
                    if filename.endswith('.pdf'):
                        content = extract_text(file_path)
                        print(f"DEBUG: Extracted PDF content length for {filename}: {len(content)}")
                    else:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            print(f"DEBUG: Extracted TXT content length for {filename}: {len(content)}")
                except Exception as e:
                    print(f"ERROR: Could not read {filename}: {e}")
                    continue # Skip to the next file if an error occurs

                if content: # Only process if content is not empty
                    # Extract paper metadata
                    title = self._extract_paper_title(content)
                    authors = self._extract_paper_authors(content)
                    abstract = self._extract_paper_abstract(content)

                    # Process text with enhanced NLP
                    doc = self.nlp(content)
                    bert_embeddings = self.extract_contextual_embeddings(content)
                    sentence_embeddings = self.sentence_model.encode([content])[0]
                    advanced_keywords = self.extract_advanced_keywords(content)
                    entities = self._extract_entities_with_links(doc)

                    self.research_papers[paper_id] = {
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'full_text': content,
                        'bert_embeddings': bert_embeddings,
                        'sentence_embeddings': sentence_embeddings,
                        'keywords': self._extract_keywords(doc),
                        'advanced_keywords': advanced_keywords,
                        'entities': entities,
                        'publication_year': self._extract_latest_publication_year(content) # Use latest year
                    }
                    paper_texts.append(content)
                    print(f"Processed paper: {title} by {', '.join(authors[:2])}...")
                else:
                    print(f"DEBUG: No content found/extracted for {filename}")

        print(f"DEBUG: Total number of texts passed to _build_topic_model: {len(paper_texts)}")
        if not paper_texts:
            print("DEBUG: paper_texts is empty before calling _build_topic_model. Check your input files and their content.")

        self._build_topic_model(paper_texts)


    def analyze_grant_proposal(self, proposal_text):
        """Backward compatible method that calls enhanced version"""
        return self.analyze_grant_proposal_enhanced(proposal_text)

    def predict_collaborations(self, grant_proposal_text, top_n=5, team_sizes=[2, 3], embedding_weight=0.7, keyword_weight=0.3, entity_weight=0.0, topic_weight=0.0, collaboration_weight=0.1, diversity_weight=0.2, use_ml=True):
        """
        Main prediction method that uses ML when available
        Parameters match the original method for backward compatibility
        """
        # Update optimal weights if provided
        if embedding_weight != 0.7 or keyword_weight != 0.3: # Check if default weights are overridden
            # Recalculate weights based on provided values, ensuring sum is 1 if all are provided
            total_explicit_weights = embedding_weight + keyword_weight + entity_weight + topic_weight + self.optimal_weights['knowledge_graph_weight']
            if total_explicit_weights > 0: # Avoid division by zero
                self.optimal_weights['embedding_weight'] = embedding_weight / total_explicit_weights
                self.optimal_weights['keyword_weight'] = keyword_weight / total_explicit_weights
                self.optimal_weights['entity_weight'] = entity_weight / total_explicit_weights
                self.optimal_weights['topic_weight'] = topic_weight / total_explicit_weights
            else: # Fallback if explicit weights sum to zero, keep current knowledge_graph_weight
                self.optimal_weights['embedding_weight'] = 0.0
                self.optimal_weights['keyword_weight'] = 0.0
                self.optimal_weights['entity_weight'] = 0.0
                self.optimal_weights['topic_weight'] = 0.0

        self.optimal_weights['collaboration_weight'] = collaboration_weight
        self.optimal_weights['diversity_weight'] = diversity_weight

        # Use enhanced prediction
        return self.predict_collaborations_with_ml(
            grant_proposal_text,
            top_n=top_n,
            team_sizes=team_sizes,
            use_ml=use_ml and self.ml_model is not None
        )

    def visualize_collaboration_network(self, output_path="collaboration_network.png", highlight_faculty=None, color_by_department=True, interactive=False):
        """
        Visualize the collaboration network of faculty.
        
        Parameters:
        output_path (str): Path to save the visualization image or HTML.
        highlight_faculty (list): Optional list of faculty IDs to highlight.
        color_by_department (bool): If True, nodes are colored by department.
        interactive (bool): If True, generates an interactive HTML visualization using pyvis.
        """
        if interactive and PYVIS_AVAILABLE:
            return self.visualize_collaboration_network_interactive(output_path.replace('.png', '.html'), highlight_faculty)
        
        # Fallback to static matplotlib visualization if interactive is not requested or not available
        print(f"Generating static collaboration network visualization to {output_path}")
        
        # Create a subgraph containing only faculty nodes
        faculty_nodes = [node for node, attrs in self.collaboration_graph.nodes(data=True) if attrs.get('type') == 'faculty']
        if not faculty_nodes:
            print("No faculty nodes found to visualize.")
            return None
            
        faculty_graph = self.collaboration_graph.subgraph(faculty_nodes)
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(faculty_graph, k=0.15, iterations=50) # Spring layout for better spread

        # Node colors
        node_colors = []
        if color_by_department:
            departments = [self.faculty_profiles[node]['department'] for node in faculty_graph.nodes()]
            unique_departments = list(set(departments))
            colors_map = plt.cm.get_cmap('tab20', len(unique_departments))
            for node in faculty_graph.nodes():
                dept_index = unique_departments.index(self.faculty_profiles[node]['department'])
                node_colors.append(colors_map(dept_index))
        else:
            node_colors = ['skyblue'] * len(faculty_graph.nodes())

        # Node sizes based on collaboration strength
        node_sizes = [self._calculate_collaboration_strength(node) * 100 + 300 for node in faculty_graph.nodes()]
        
        # Edge widths based on collaboration weight
        edge_weights = [d['weight'] * 2 for u, v, d in faculty_graph.edges(data=True)]
        edge_colors = ['#cccccc'] * len(faculty_graph.edges())

        # Highlight specified faculty
        if highlight_faculty:
            for i, node in enumerate(faculty_graph.nodes()):
                if node in highlight_faculty:
                    node_colors[i] = 'red' # Highlight color
                    node_sizes[i] += 200 # Make highlighted nodes larger
            for i, edge in enumerate(faculty_graph.edges()):
                if edge[0] in highlight_faculty and edge[1] in highlight_faculty:
                    edge_colors[i] = 'orange' # Highlight edges between highlighted faculty


        nx.draw_networkx_nodes(
            faculty_graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=1
        )
        nx.draw_networkx_edges(
            faculty_graph, pos,
            width=edge_weights,
            alpha=0.6,
            edge_color=edge_colors
        )

        # Create readable labels
        labels = {}
        for node in faculty_nodes:
            full_name = self.faculty_profiles[node]['name']
            name_parts = full_name.split()
            if len(name_parts) > 1:
                first_initial = name_parts[0][0]
                last_name = name_parts[-1]
                labels[node] = f"{first_initial}. {last_name}"
            else:
                labels[node] = full_name # Use full name if only one part

        nx.draw_networkx_labels(
            faculty_graph, pos,
            labels=labels,
            font_size=11,
            font_weight='bold',
            font_color='black'
        )
        
        # Add legend if color_by_department
        if color_by_department:
            unique_departments = list(set([self.faculty_profiles[node]['department'] for node in faculty_graph.nodes()]))
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=dept,
                                markersize=10, markerfacecolor=colors_map(i))
                       for i, dept in enumerate(unique_departments)]
            plt.legend(handles=handles, title="Departments", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title('Faculty Collaboration Network', size=15)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close()
        print(f"Static network visualization saved to {output_path}")
        return output_path
    
    def visualize_collaboration_network(self, output_path="collaboration_network.png", highlight_faculty=None, 
                                        color_by_department=True, interactive=False):
        """
        Visualize the collaboration network of faculty.
        
        Parameters:
        output_path (str): Path to save the visualization image or HTML.
        highlight_faculty (list): Optional list of faculty IDs to highlight.
        color_by_department (bool): If True, nodes are colored by department.
        interactive (bool): If True, generates an interactive HTML visualization using pyvis.
        
        Returns:
        str: Path to the saved visualization
        """
        # Debug logging
        print(f"DEBUG: visualize_collaboration_network called with:")
        print(f"  output_path type: {type(output_path)}, value: {output_path}")
        print(f"  highlight_faculty type: {type(highlight_faculty)}, value: {highlight_faculty}")
        print(f"  color_by_department type: {type(color_by_department)}, value: {color_by_department}")
        print(f"  interactive type: {type(interactive)}, value: {interactive}")
        
        # Ensure output_path is a string
        if not isinstance(output_path, str):
            raise TypeError(f"output_path must be a string, got {type(output_path).__name__}: {output_path}")
        
        if interactive and PYVIS_AVAILABLE:
            # For interactive visualization, ensure .html extension
            if not output_path.endswith('.html'):
                # Replace any existing extension with .html
                base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
                output_path = base_path + '.html'
            return self.visualize_collaboration_network_interactive(output_path, highlight_faculty)
        
        # Fallback to static matplotlib visualization
        print(f"Generating static collaboration network visualization to {output_path}")
        
        # Ensure .png extension for static
        if not output_path.endswith('.png'):
            base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
            output_path = base_path + '.png'
        
        # Create a subgraph containing only faculty nodes
        faculty_nodes = [node for node, attrs in self.collaboration_graph.nodes(data=True) 
                        if attrs.get('type') == 'faculty']
        if not faculty_nodes:
            print("No faculty nodes found to visualize.")
            return None
            
        faculty_graph = self.collaboration_graph.subgraph(faculty_nodes)
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(faculty_graph, k=0.15, iterations=50)

        # Node colors
        node_colors = []
        unique_departments = []
        
        if color_by_department:
            departments = [self.faculty_profiles.get(node, {}).get('department', 'Unknown') 
                          for node in faculty_graph.nodes()]
            unique_departments = list(set(departments))
            colors_map = plt.cm.get_cmap('tab20', len(unique_departments))
            
            for node in faculty_graph.nodes():
                dept = self.faculty_profiles.get(node, {}).get('department', 'Unknown')
                dept_index = unique_departments.index(dept)
                node_colors.append(colors_map(dept_index))
        else:
            node_colors = ['skyblue'] * len(faculty_graph.nodes())

        # Node sizes based on collaboration strength
        node_sizes = [self._calculate_collaboration_strength(node) * 100 + 300 
                     for node in faculty_graph.nodes()]
        
        # Edge widths based on collaboration weight
        edge_weights = [d.get('weight', 1) * 2 
                       for u, v, d in faculty_graph.edges(data=True)]
        edge_colors = ['#cccccc'] * len(faculty_graph.edges())

        # Highlight specified faculty
        if highlight_faculty:
            for i, node in enumerate(faculty_graph.nodes()):
                if node in highlight_faculty:
                    node_colors[i] = 'red'
                    node_sizes[i] += 200
            
            for i, (u, v) in enumerate(faculty_graph.edges()):
                if u in highlight_faculty and v in highlight_faculty:
                    edge_colors[i] = 'orange'

        # Draw nodes
        nx.draw_networkx_nodes(
            faculty_graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=1
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            faculty_graph, pos,
            width=edge_weights,
            alpha=0.6,
            edge_color=edge_colors
        )

        # Create readable labels
        labels = {}
        for node in faculty_nodes:
            if node in self.faculty_profiles:
                full_name = self.faculty_profiles[node]['name']
                name_parts = full_name.split()
                if len(name_parts) > 1:
                    first_initial = name_parts[0][0]
                    last_name = name_parts[-1]
                    labels[node] = f"{first_initial}. {last_name}"
                else:
                    labels[node] = full_name
            else:
                labels[node] = node

        # Draw labels
        nx.draw_networkx_labels(
            faculty_graph, pos,
            labels=labels,
            font_size=11,
            font_weight='bold',
            font_color='black'
        )
        
        # Add legend if color_by_department
        if color_by_department and len(unique_departments) > 0:
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=dept,
                                markersize=10, markerfacecolor=colors_map(i))
                      for i, dept in enumerate(unique_departments)]
            plt.legend(handles=handles, title="Departments", 
                      bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title('Faculty Collaboration Network', size=15)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close()
        print(f"Static network visualization saved to {output_path}")
        return output_path

    def visualize_collaboration_network_interactive(self, output_path="collaboration_network.html", 
                                                  highlight_faculty=None):
        """
        Create an interactive collaboration network visualization using pyvis
        
        Parameters:
        output_path (str): Path to save the HTML file
        highlight_faculty (list): Faculty IDs to highlight
        
        Returns:
        str: Path to the saved visualization
        """
        if not PYVIS_AVAILABLE:
            print("pyvis is not available. Cannot generate interactive visualization.")
            return None

        try:
            from pyvis.network import Network
            
            # Create network with proper configuration
            net = Network(
                height="750px", 
                width="100%", 
                bgcolor="#222222", 
                font_color="white"
            )
            
            # Configure physics
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "repulsion": {
                  "nodeDistance": 200,
                  "centralGravity": 0.05,
                  "springLength": 150,
                  "springConstant": 0.05,
                  "damping": 0.09
                }
              }
            }
            """)

            # Check if we have faculty data
            if not self.faculty_profiles:
                print("No faculty profiles available for visualization")
                return None

            # Add nodes
            for faculty_id, profile in self.faculty_profiles.items():
                # Prepare node information
                title = (f"<b>{profile['name']}</b><br>"
                        f"Department: {profile['department']}<br>"
                        f"Research Interests: {profile.get('research_interests', 'N/A')}")
                
                # Set node color
                color = "#007bff"  # Default blue
                if highlight_faculty and faculty_id in highlight_faculty:
                    color = "#dc3545"  # Red for highlighted
                
                # Calculate node size based on collaboration strength
                size = self._calculate_collaboration_strength(faculty_id) * 5 + 15 
                
                # Add node to network
                net.add_node(
                    faculty_id, 
                    label=profile['name'], 
                    title=title, 
                    color=color, 
                    size=size
                )
            
            # Add edges
            for u, v, attrs in self.collaboration_graph.edges(data=True):
                if u in self.faculty_profiles and v in self.faculty_profiles:
                    weight = attrs.get('weight', 1)
                    title = f"Collaboration Strength: {weight:.2f}"
                    
                    # Set edge color
                    color = "#6c757d"  # Default gray
                    if highlight_faculty and u in highlight_faculty and v in highlight_faculty:
                        color = "#ffc107"  # Yellow for highlighted connections
                    
                    # Add edge to network
                    net.add_edge(
                        u, v, 
                        value=weight, 
                        title=title, 
                        width=weight, 
                        color=color
                    )

            # Save the visualization using the correct method
            try:
                # Try the newer method first
                net.save_graph(output_path)
            except AttributeError:
                try:
                    # Fallback to older method
                    net.show(output_path)
                except AttributeError:
                    # Last resort - write HTML manually
                    html_content = net.generate_html()
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
            
            print(f"Interactive network visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating interactive visualization: {str(e)}")
            return None

    def visualize_collaboration_network(self, output_path="collaboration_network.png", highlight_faculty=None, 
                                        color_by_department=True, interactive=False):
        """
        Visualize the collaboration network of faculty.
        
        Parameters:
        output_path (str): Path to save the visualization image or HTML.
        highlight_faculty (list): Optional list of faculty IDs to highlight.
        color_by_department (bool): If True, nodes are colored by department.
        interactive (bool): If True, generates an interactive HTML visualization using pyvis.
        
        Returns:
        str: Path to the saved visualization
        """
        
        if interactive and PYVIS_AVAILABLE:
            # For interactive visualization, ensure .html extension
            if not output_path.endswith('.html'):
                base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
                output_path = base_path + '.html'
            return self.visualize_collaboration_network_interactive(output_path, highlight_faculty)
        
        # Fallback to static matplotlib visualization
        print(f"Generating static collaboration network visualization to {output_path}")
        
        # Ensure .png extension for static
        if not output_path.endswith('.png'):
            base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
            output_path = base_path + '.png'
        
        try:
            # Create a subgraph containing only faculty nodes
            faculty_nodes = [node for node, attrs in self.collaboration_graph.nodes(data=True) 
                            if attrs.get('type') == 'faculty']
            
            if not faculty_nodes:
                print("No faculty nodes found to visualize.")
                return None
                
            faculty_graph = self.collaboration_graph.subgraph(faculty_nodes)
            
            plt.figure(figsize=(12, 10))
            
            # Use different layout algorithms based on network size
            if len(faculty_nodes) > 20:
                pos = nx.spring_layout(faculty_graph, k=0.5, iterations=50)
            else:
                pos = nx.spring_layout(faculty_graph, k=1, iterations=50)

            # Node colors
            node_colors = []
            unique_departments = []
            
            if color_by_department and self.faculty_profiles:
                departments = [self.faculty_profiles.get(node, {}).get('department', 'Unknown') 
                              for node in faculty_graph.nodes()]
                unique_departments = list(set(departments))
                
                if len(unique_departments) > 20:
                    colors_map = plt.cm.get_cmap('tab20', 20)  # Limit to 20 colors
                    unique_departments = unique_departments[:20]
                else:
                    colors_map = plt.cm.get_cmap('tab20', len(unique_departments))
                
                for node in faculty_graph.nodes():
                    dept = self.faculty_profiles.get(node, {}).get('department', 'Unknown')
                    if dept in unique_departments:
                        dept_index = unique_departments.index(dept)
                        node_colors.append(colors_map(dept_index))
                    else:
                        node_colors.append('gray')
            else:
                node_colors = ['skyblue'] * len(faculty_graph.nodes())

            # Node sizes based on collaboration strength
            node_sizes = []
            for node in faculty_graph.nodes():
                size = self._calculate_collaboration_strength(node) * 100 + 300
                node_sizes.append(min(size, 1000))  # Cap maximum size
            
            # Edge widths based on collaboration weight
            edge_weights = []
            for u, v, d in faculty_graph.edges(data=True):
                weight = d.get('weight', 1)
                edge_weights.append(min(weight * 2, 10))  # Cap maximum width
            
            edge_colors = ['#cccccc'] * len(faculty_graph.edges())

            # Highlight specified faculty
            if highlight_faculty:
                for i, node in enumerate(faculty_graph.nodes()):
                    if node in highlight_faculty:
                        node_colors[i] = 'red'
                        node_sizes[i] += 200
                
                for i, (u, v) in enumerate(faculty_graph.edges()):
                    if u in highlight_faculty and v in highlight_faculty:
                        edge_colors[i] = 'orange'

            # Draw nodes
            nx.draw_networkx_nodes(
                faculty_graph, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.9,
                edgecolors='black',
                linewidths=1
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                faculty_graph, pos,
                width=edge_weights,
                alpha=0.6,
                edge_color=edge_colors
            )

            # Create readable labels
            labels = {}
            for node in faculty_nodes:
                if node in self.faculty_profiles:
                    full_name = self.faculty_profiles[node]['name']
                    name_parts = full_name.split()
                    if len(name_parts) > 1:
                        first_initial = name_parts[0][0]
                        last_name = name_parts[-1]
                        labels[node] = f"{first_initial}. {last_name}"
                    else:
                        labels[node] = full_name
                else:
                    labels[node] = str(node)[:10]  # Truncate long IDs

            # Draw labels with smaller font for large networks
            font_size = 8 if len(faculty_nodes) > 15 else 11
            
            nx.draw_networkx_labels(
                faculty_graph, pos,
                labels=labels,
                font_size=font_size,
                font_weight='bold',
                font_color='black'
            )
            
            # Add legend if color_by_department and reasonable number of departments
            if color_by_department and len(unique_departments) > 0 and len(unique_departments) <= 10:
                handles = [plt.Line2D([0], [0], marker='o', color='w', label=dept,
                                    markersize=10, markerfacecolor=colors_map(i))
                          for i, dept in enumerate(unique_departments)]
                plt.legend(handles=handles, title="Departments", 
                          bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.title('Faculty Collaboration Network', size=15)
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Static network visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating static visualization: {str(e)}")
            return None
    

    def generate_faculty_profile_cards(self, output_dir="faculty_profile_cards"):
        """
        Generates individual faculty profile cards with key information and visualizations.
        
        Parameters:
        output_dir (str): Directory to save profile cards
        
        Returns:
        list: Paths to generated profile cards
        """
        os.makedirs(output_dir, exist_ok=True)
        profile_paths = []

        for faculty_id, profile in self.faculty_profiles.items():
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # Header
            ax_header = fig.add_subplot(gs[0, :])
            ax_header.axis('off')
            ax_header.text(0.5, 0.8, profile['name'], fontsize=20, weight='bold', ha='center')
            ax_header.text(0.5, 0.4, profile['department'], fontsize=14, ha='center')
            ax_header.text(0.5, 0.1, f"Research Focus: {profile.get('research_interests', 'N/A')}", fontsize=10, ha='center', wrap=True)

            # Publication timeline
            ax_timeline = fig.add_subplot(gs[1, :2])
            pub_years = profile.get('publication_years', [])
            if pub_years:
                year_counts = pd.Series(pub_years).value_counts().sort_index()
                ax_timeline.bar(year_counts.index, year_counts.values, color='skyblue', edgecolor='navy')
                ax_timeline.set_xlabel('Year')
                ax_timeline.set_ylabel('Publications')
                ax_timeline.set_title('Publication Timeline')
            else:
                ax_timeline.text(0.5, 0.5, 'No publication data available', ha='center', va='center', transform=ax_timeline.transAxes)

            # Keyword cloud / Top Keywords
            ax_keywords = fig.add_subplot(gs[1, 2])
            keywords = dict(profile.get('advanced_keywords', [])[:20]) # Top 20 keywords
            if keywords:
                # Create a simple bar chart of top keywords
                sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
                kw_names = [k[0] for k in sorted_keywords]
                kw_scores = [k[1] for k in sorted_keywords]
                ax_keywords.barh(kw_names, kw_scores, color='lightgreen', edgecolor='darkgreen')
                ax_keywords.set_xlabel('Score')
                ax_keywords.set_title('Top Keywords')
                ax_keywords.invert_yaxis() # Top keyword at the top
            else:
                ax_keywords.text(0.5, 0.5, 'No keywords available', ha='center', va='center', transform=ax_keywords.transAxes)
            
            # Expertise Evolution
            ax_evolution = fig.add_subplot(gs[2, :])
            evolution_data = self._analyze_expertise_evolution(profile)
            ax_evolution.axis('off')
            ax_evolution.text(0.05, 0.9, "Expertise Evolution:", fontsize=12, weight='bold')
            ax_evolution.text(0.05, 0.6, f"Trend: {evolution_data['trend'].capitalize()}", fontsize=10)
            ax_evolution.text(0.05, 0.3, f"Recent Activity (last 5 years): {'Yes' if evolution_data['recent_activity'] else 'No'}", fontsize=10)

            plt.tight_layout()
            
            # Save card
            card_path = os.path.join(output_dir, f"{faculty_id}_profile_card.png")
            plt.savefig(card_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            profile_paths.append(card_path)
            print(f"Generated profile card for {profile['name']}")
        
        return profile_paths
    
    def generate_collaboration_report_pdf(self, recommendations, output_path="collaboration_report.pdf"):
        """
        Generates a detailed PDF report of collaboration recommendations.
        
        Parameters:
        recommendations (list): List of recommended collaborations.
        output_path (str): Path to save the PDF report
        
        Returns:
        str: Path to the generated report
        """
        if not REPORTLAB_AVAILABLE:
            print("ReportLab is not available. Generating a simplified text report instead.")
            # Fallback to text report
            with open(output_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
                f.write("FACULTY COLLABORATION RECOMMENDATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Number of recommendations: {len(recommendations)}\n\n")
                
                for i, rec in enumerate(recommendations):
                    f.write(f"RECOMMENDATION #{i+1}\n")
                    f.write(f"Team Size: {rec['team_size']}\n")
                    f.write(f"Combined Score: {rec['combined_score']:.3f}\n")
                    f.write(f"Team Members:\n")
                    for j, name in enumerate(rec['faculty_names']):
                        dept = rec['departments'][j] if j < len(rec['departments']) else 'N/A'
                        f.write(f" - {name} ({dept})\n")
                    f.write("\n")
                    f.write("Synergy Reasons:\n")
                    for reason in rec['synergy_reasons']:
                        f.write(f"  • {reason}\n")
                    f.write("\n" + "-" * 40 + "\n\n") # Separator for recommendations
            print(f"Text report generated: {output_path.replace('.pdf', '.txt')}")
            return output_path.replace('.pdf', '.txt')

        # Original PDF generation code
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30
        )
        story.append(Paragraph("Faculty Collaboration Recommendation Report", title_style))
        story.append(Spacer(1, 20))

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        summary_text = f"""
        This report presents {len(recommendations)} recommended faculty teams for the grant proposal. The recommendations are based on advanced machine learning analysis, knowledge graph matching, and comprehensive evaluation of expertise alignment, collaboration history, and team synergy.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Grant Analysis Summary (Placeholder - needs actual grant analysis to be passed)
        # story.append(Paragraph("Grant Proposal Analysis Summary", styles['Heading1']))
        # story.append(Paragraph("Brief summary of grant keywords, entities, and requirements.", styles['Normal']))
        # story.append(Spacer(1, 20))

        # Recommendations Section
        story.append(Paragraph("Recommended Teams", styles['Heading1']))
        story.append(Spacer(1, 10))

        for i, rec in enumerate(recommendations):
            story.append(Paragraph(f"Recommendation #{i+1} (Team Size: {rec['team_size']})", styles['Heading2']))
            story.append(Paragraph(f"<b>Combined Score: {rec['combined_score']:.3f}</b>", styles['Normal']))
            story.append(Spacer(1, 10))

            # Team Members Table
            story.append(Paragraph("Team Members:", styles['Heading3']))
            faculty_data = [['Name', 'Department', 'Individual Score', 'Top Expertise']]
            for j, faculty_id in enumerate(rec['faculty']):
                name = rec['faculty_names'][j]
                dept = rec['departments'][j]
                score = rec['individual_scores'][j]
                # Get top keywords for this faculty
                expertise = 'N/A'
                if faculty_id in self.faculty_profiles:
                    keywords = [kw for kw, _ in self.faculty_profiles[faculty_id].get('advanced_keywords', [])[:3]]
                    expertise = ', '.join(keywords) if keywords else 'N/A'
                faculty_data.append([name, dept, f"{score:.3f}", expertise])
            
            faculty_table = Table(faculty_data, colWidths=[150, 120, 80, 180])
            faculty_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            story.append(faculty_table)
            story.append(Spacer(1, 10))

            # Team synergy
            story.append(Paragraph("Team Synergy Analysis:", styles['Heading3']))
            for reason in rec['synergy_reasons'][:5]: # Limit to top 5 reasons
                story.append(Paragraph(f"• {reason}", styles['Normal']))
            story.append(Spacer(1, 10))

            # Detailed metrics
            metrics_data = [
                ['Metric', 'Score'],
                ['Base Expertise Match', f"{rec['base_score']:.3f}"],
                ['Collaboration Synergy', f"{rec['collaboration_bonus']:.3f}"],
                ['Diversity Bonus', f"{rec['diversity_score']:.3f}"],
                ['Complementarity', f"{rec['complementarity_score']:.3f}"]
            ]
            metrics_table = Table(metrics_data, colWidths=[150, 80])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#a0a0a0'))
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 20)) # Space after each recommendation
            story.append(PageBreak()) # Start new page for next recommendation

        doc.build(story)
        print(f"PDF report generated: {output_path}")
        return output_path
def predict_collaborations_debug(self, grant_proposal_text, top_n=5, team_sizes=[2, 3], embedding_weight=0.7, keyword_weight=0.3, entity_weight=0.0, topic_weight=0.0, collaboration_weight=0.1, diversity_weight=0.2, use_ml=True):
    """
    Debug version of predict_collaborations with extensive logging
    """
    print("\n=== DEBUG: predict_collaborations called ===")
    print(f"Parameters received:")
    print(f"  grant_proposal_text: {type(grant_proposal_text)}, length: {len(grant_proposal_text) if isinstance(grant_proposal_text, str) else 'N/A'}")
    print(f"  top_n: {type(top_n)}, value: {top_n}")
    print(f"  team_sizes: {type(team_sizes)}, value: {team_sizes}")
    print(f"  use_ml: {type(use_ml)}, value: {use_ml}")
    
    try:
        # Update optimal weights if provided
        if embedding_weight != 0.7 or keyword_weight != 0.3:
            total_explicit_weights = embedding_weight + keyword_weight + entity_weight + topic_weight + self.optimal_weights['knowledge_graph_weight']
            if total_explicit_weights > 0:
                self.optimal_weights['embedding_weight'] = embedding_weight / total_explicit_weights
                self.optimal_weights['keyword_weight'] = keyword_weight / total_explicit_weights
                self.optimal_weights['entity_weight'] = entity_weight / total_explicit_weights
                self.optimal_weights['topic_weight'] = topic_weight / total_explicit_weights
            else:
                self.optimal_weights['embedding_weight'] = 0.0
                self.optimal_weights['keyword_weight'] = 0.0
                self.optimal_weights['entity_weight'] = 0.0
                self.optimal_weights['topic_weight'] = 0.0

        self.optimal_weights['collaboration_weight'] = collaboration_weight
        self.optimal_weights['diversity_weight'] = diversity_weight

        print("\nDEBUG: Calling predict_collaborations_with_ml...")
        
        # Use enhanced prediction
        result = self.predict_collaborations_with_ml(
            grant_proposal_text,
            top_n=top_n,
            team_sizes=team_sizes,
            use_ml=use_ml and self.ml_model is not None
        )
        
        print(f"\nDEBUG: predict_collaborations_with_ml returned {len(result) if isinstance(result, list) else 'non-list'} results")
        return result
        
    except Exception as e:
        print(f"\nDEBUG ERROR in predict_collaborations: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def _find_optimal_combinations_ml_debug(self, faculty_matches, proposal_analysis, top_n=5, team_sizes=[2, 3]):
    """
    Debug version of _find_optimal_combinations_ml
    """
    print("\n=== DEBUG: _find_optimal_combinations_ml called ===")
    print(f"  faculty_matches: {type(faculty_matches)}, length: {len(faculty_matches) if isinstance(faculty_matches, dict) else 'N/A'}")
    print(f"  top_n: {top_n}")
    print(f"  team_sizes: {team_sizes}")
    
    # Sort faculty by match score
    sorted_faculty = sorted(faculty_matches.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    # Limit pool size for computational efficiency
    max_team_size = max(team_sizes)
    pool_size = min(len(sorted_faculty), max(15, 3 * max_team_size))
    top_faculty = sorted_faculty[:pool_size]
    
    print(f"\nDEBUG: Pool size: {pool_size}, top_faculty length: {len(top_faculty)}")
    
    combinations = []
    from itertools import combinations as itertools_combinations
    
    for size in team_sizes:
        if size > len(top_faculty):
            continue
        
        print(f"\nDEBUG: Generating combinations for team size {size}")
        
        for combo_indices in itertools_combinations(range(len(top_faculty)), size):
            combo = [top_faculty[i] for i in combo_indices]
            faculty_ids = [f[0] for f in combo]
            
            # Extract features for ML model
            features = self._extract_team_features(faculty_ids, proposal_analysis['text'])
            features_scaled = self.feature_scaler.transform([features])
            
            # Predict score using ML model
            ml_score = self.ml_model.predict(features_scaled)[0]
            
            # Calculate additional scores
            print(f"\nDEBUG: Calculating combination score for team: {faculty_ids}")
            score_info = self._calculate_combination_score_enhanced(combo, proposal_analysis)
            
            # Combine ML score with other factors
            combined_score = 0.7 * ml_score + 0.3 * score_info['rule_based_score']
            
            combinations.append({
                'faculty': faculty_ids,
                'faculty_names': [f[1]['faculty_name'] for f in combo],
                'departments': [f[1]['department'] for f in combo],
                'individual_scores': [f[1]['overall_score'] for f in combo],
                'individual_explanations': [f[1]['match_explanation'] for f in combo],
                'ml_score': ml_score,
                'combined_score': combined_score,
                'base_score': score_info['base_score'],
                'collaboration_bonus': score_info['collaboration_bonus'],
                'diversity_score': score_info['diversity_score'],
                'complementarity_score': score_info['complementarity_score'],
                'synergy_reasons': score_info['synergy_reasons'],
                'collaboration_network': score_info['collaboration_network'],
                'expertise_coverage': score_info['expertise_coverage'],
                'team_size': size
            })
    
    print(f"\nDEBUG: Generated {len(combinations)} total combinations")
    
    # Sort combinations by combined score
    sorted_combinations = sorted(combinations, key=lambda x: x['combined_score'], reverse=True)
    return sorted_combinations[:top_n]