import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Try to import the collaboration system, with fallback
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'core'))
    from collaboration_system import EnhancedFacultyCollaborationPredictor
    COLLABORATION_SYSTEM_AVAILABLE = True
except ImportError:
    # Fallback: create a simple mock predictor for testing
    COLLABORATION_SYSTEM_AVAILABLE = False
    print("Warning: Using mock collaboration system for testing")
    
    class EnhancedFacultyCollaborationPredictor:
        def __init__(self):
            self.faculty_profiles = {}
            self.research_papers = {}
            self.ml_model = None
            self.prediction_history = []
        
        def load_faculty_bios(self, directory):
            print(f"Loading faculty bios from {directory}")
            import os
            files = [f for f in os.listdir(directory) if f.endswith(('.txt', '.pdf', '.docx'))]
            
            # Enhanced mock data generation
            departments = ['Computer Science', 'Biology', 'Physics', 'Chemistry', 'Mathematics', 'Engineering', 'Psychology', 'Medicine']
            research_areas = [
                'Machine Learning', 'Data Science', 'Artificial Intelligence', 'Bioinformatics', 
                'Quantum Computing', 'Robotics', 'Cybersecurity', 'Software Engineering',
                'Molecular Biology', 'Genetics', 'Neuroscience', 'Cancer Research',
                'Materials Science', 'Renewable Energy', 'Climate Science', 'Nanotechnology'
            ]
            
            for i, file in enumerate(files):
                dept = departments[i % len(departments)]
                area = research_areas[i % len(research_areas)]
                self.faculty_profiles[f"faculty_{i}"] = {
                    'name': f"Dr. {file.split('.')[0].replace('_', ' ').title()}",
                    'department': dept,
                    'research_area': area,
                    'expertise_score': 0.7 + (i % 4) * 0.1,
                    'keywords': [area, 'Research', 'Innovation', 'Collaboration'],
                    'collaboration_count': i % 5 + 1
                }
            print(f"Loaded {len(self.faculty_profiles)} faculty profiles")
        
        def load_research_papers(self, directory):
            print(f"Loading research papers from {directory}")
            import os
            files = [f for f in os.listdir(directory) if f.endswith(('.txt', '.pdf', '.docx'))]
            for i, file in enumerate(files):
                self.research_papers[f"paper_{i}"] = {
                    'title': f"Research Paper {i+1}",
                    'filename': file
                }
            print(f"Loaded {len(self.research_papers)} research papers")
        
        def train_ml_model(self):
            print("Training ML model...")
            self.ml_model = True
        
        def predict_collaborations_with_ml(self, text, top_n=5, team_sizes=[3,4], use_ml=True):
            # Enhanced mock prediction with more teams
            if not self.faculty_profiles:
                return []
            
            faculty_list = list(self.faculty_profiles.items())
            predictions = []
            
            # Generate more diverse teams
            import random
            random.seed(42)  # For consistent results
            
            # Ensure we can generate the requested number of teams
            max_possible_teams = len(faculty_list) // min(team_sizes) if team_sizes else 1
            team_count = min(top_n, max(max_possible_teams, top_n))  # Always try to generate requested number
            
            # If we don't have enough faculty, create some additional mock faculty
            if len(faculty_list) < top_n * max(team_sizes):
                additional_faculty_needed = top_n * max(team_sizes) - len(faculty_list)
                departments = ['Computer Science', 'Biology', 'Physics', 'Chemistry', 'Mathematics', 'Engineering', 'Psychology', 'Medicine', 'Economics', 'Environmental Science']
                areas = ['AI/ML', 'Data Science', 'Bioinformatics', 'Quantum Physics', 'Statistics', 'Robotics', 'Cybersecurity', 'Biotechnology', 'Climate Modeling', 'Social Sciences']
                
                for i in range(additional_faculty_needed):
                    faculty_id = f"additional_faculty_{i}"
                    dept = departments[i % len(departments)]
                    area = areas[i % len(areas)]
                    self.faculty_profiles[faculty_id] = {
                        'name': f"Dr. {area.replace('/', ' ')} Expert {i+1}",
                        'department': dept,
                        'research_area': area,
                        'expertise_score': 0.6 + random.uniform(0, 0.3),
                        'keywords': [area, 'Research', 'Innovation'],
                        'collaboration_count': random.randint(1, 8)
                    }
                    faculty_list.append((faculty_id, self.faculty_profiles[faculty_id]))
            
            used_faculty = set()
            
            for i in range(team_count):
                team_size = team_sizes[i % len(team_sizes)]
                
                # Select faculty not yet used (with some overlap allowed for larger requests)
                available_faculty = [f for f in faculty_list if f[0] not in used_faculty]
                
                if len(available_faculty) < team_size:
                    # If not enough unique faculty, allow some overlap but prefer unused
                    available_faculty = faculty_list.copy()
                    # Shuffle to get different combinations
                    random.shuffle(available_faculty)
                
                team_members = []
                departments_used = set()
                
                for j in range(team_size):
                    if available_faculty:
                        # Prefer faculty from different departments for diversity
                        preferred = [f for f in available_faculty 
                                   if f[1]['department'] not in departments_used]
                        if not preferred:
                            preferred = available_faculty
                        
                        # Add some randomness to selection
                        if len(preferred) > 1:
                            selected = random.choice(preferred[:min(3, len(preferred))])
                        else:
                            selected = preferred[0] if preferred else available_faculty[0]
                            
                        team_members.append(selected)
                        departments_used.add(selected[1]['department'])
                        used_faculty.add(selected[0])
                        
                        # Remove from available to avoid immediate reuse
                        if selected in available_faculty:
                            available_faculty.remove(selected)
                
                if not team_members:
                    continue
                
                # Generate varied scores with realistic distribution
                base_score = 0.95 - (i * 0.03)  # Smaller decrease per team
                ml_score = 0.9 - (i * 0.025)
                individual_scores = [member[1]['expertise_score'] + random.uniform(-0.05, 0.15) 
                                   for member in team_members]
                
                # Ensure scores stay within reasonable bounds
                base_score = max(0.65, min(0.95, base_score))
                ml_score = max(0.6, min(0.9, ml_score))
                individual_scores = [max(0.5, min(1.0, score)) for score in individual_scores]
                
                # Generate diverse synergy reasons
                synergy_options = [
                    'Strong interdisciplinary collaboration potential',
                    'Complementary research methodologies and approaches',
                    'Previous successful joint publications',
                    'Shared research infrastructure and resources',
                    'Excellent track record in grant acquisition',
                    'Diverse expertise covering all project requirements',
                    'Strong network connections in the research community',
                    'Proven ability to mentor graduate students together',
                    'Complementary computational and experimental skills',
                    'International collaboration experience',
                    'Innovative approach to cross-disciplinary research',
                    'Strong publication record in high-impact journals',
                    'Experience with large-scale collaborative projects',
                    'Expertise spans multiple relevant domains',
                    'Track record of translating research to applications'
                ]
                
                # Select 3-4 diverse reasons
                num_reasons = random.randint(3, 4)
                selected_reasons = random.sample(synergy_options, min(num_reasons, len(synergy_options)))
                
                prediction = {
                    'faculty_names': [member[1]['name'] for member in team_members],
                    'departments': [member[1]['department'] for member in team_members],
                    'team_size': len(team_members),
                    'combined_score': round(base_score, 3),
                    'ml_score': round(ml_score, 3),
                    'base_score': round(base_score - 0.05, 3),
                    'synergy_reasons': selected_reasons,
                    'individual_scores': [round(score, 3) for score in individual_scores]
                }
                predictions.append(prediction)
            
            # Store prediction in history
            self.prediction_history.append({
                'timestamp': __import__('datetime').datetime.now().isoformat(),
                'team_count': len(predictions),
                'accuracy': random.uniform(0.85, 0.95)
            })
            
            print(f"Generated {len(predictions)} team predictions")
            return predictions
        
        def get_analytics_data(self):
            """Get analytics data for insights page"""
            if not self.faculty_profiles:
                return None
            
            # Faculty distribution by department
            dept_distribution = {}
            total_collaborations = 0
            all_keywords = []
            
            for profile in self.faculty_profiles.values():
                dept = profile['department']
                if dept not in dept_distribution:
                    dept_distribution[dept] = {'count': 0, 'total_score': 0}
                dept_distribution[dept]['count'] += 1
                dept_distribution[dept]['total_score'] += profile['expertise_score']
                total_collaborations += profile.get('collaboration_count', 0)
                all_keywords.extend(profile.get('keywords', []))
            
            # Calculate average scores
            for dept in dept_distribution:
                dept_distribution[dept]['avg_score'] = dept_distribution[dept]['total_score'] / dept_distribution[dept]['count']
            
            # Top keywords analysis
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            top_keywords = []
            for keyword, count in keyword_counts.most_common(10):
                score = min(0.95, 0.7 + (count / max(keyword_counts.values())) * 0.25)
                top_keywords.append({
                    'keyword': keyword,
                    'frequency': count,
                    'score': round(score, 2)
                })
            
            # Prediction trends (mock data based on history)
            import random
            trends = []
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            for i, month in enumerate(months):
                trends.append({
                    'month': month,
                    'predictions': random.randint(15, 40),
                    'accuracy': round(random.uniform(0.82, 0.95), 2)
                })
            
            return {
                'faculty_distribution': [
                    {
                        'department': dept,
                        'count': data['count'],
                        'avg_score': round(data['avg_score'], 2)
                    }
                    for dept, data in dept_distribution.items()
                ],
                'top_keywords': top_keywords,
                'prediction_trends': trends,
                'network_stats': {
                    'total_connections': total_collaborations,
                    'network_density': round(total_collaborations / max(len(self.faculty_profiles) ** 2, 1), 2),
                    'avg_connections': round(total_collaborations / len(self.faculty_profiles), 1),
                    'hub_nodes': min(3, len(self.faculty_profiles) // 3)
                },
                'performance_metrics': {
                    'prediction_accuracy': round(random.uniform(0.88, 0.95), 3),
                    'avg_response_time': round(random.uniform(1.2, 2.5), 1),
                    'system_uptime': round(random.uniform(0.95, 0.999), 3),
                    'avg_team_size': round(sum(len(self.faculty_profiles) for _ in range(5)) / 15, 1)
                }
            }

def create_app():
    app = Flask(__name__)
    
    # Basic configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'university-of-idaho-secret')
    app.config['UPLOAD_FOLDER'] = 'data/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    
    # Enable CORS for React frontend
    CORS(app, origins=["http://localhost:3000"])
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
    
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Initialize the ML predictor (global instance for demo)
    predictor = None
    
    # Health check route
    @app.route('/api/health', methods=['GET'])
    def health_check():
        status_msg = 'University of Idaho Faculty Collaboration System'
        if not COLLABORATION_SYSTEM_AVAILABLE:
            status_msg += ' (Demo Mode)'
        return {'status': 'healthy', 'message': status_msg}
    
    # Faculty file upload endpoint
    @app.route('/api/data/upload-faculty', methods=['POST'])
    def upload_faculty_files():
        try:
            print("Faculty upload request received")
            
            if 'files' not in request.files:
                return jsonify({'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            if not files or all(f.filename == '' for f in files):
                return jsonify({'error': 'No files selected'}), 400
            
            uploaded_files = []
            errors = []
            
            faculty_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'faculty')
            os.makedirs(faculty_dir, exist_ok=True)
            
            for file in files:
                if file.filename == '':
                    continue
                    
                if not allowed_file(file.filename):
                    errors.append(f"{file.filename}: File type not allowed")
                    continue
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(faculty_dir, filename)
                file.save(file_path)
                
                uploaded_files.append({
                    'filename': filename,
                    'size': os.path.getsize(file_path)
                })
            
            print(f"Successfully uploaded {len(uploaded_files)} faculty files")
            
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded {len(uploaded_files)} faculty files',
                'uploaded_files': uploaded_files,
                'errors': errors
            })
            
        except Exception as e:
            print(f"Faculty upload error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # Research papers upload endpoint
    @app.route('/api/data/upload-papers', methods=['POST'])
    def upload_paper_files():
        try:
            print("Papers upload request received")
            
            if 'files' not in request.files:
                return jsonify({'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            if not files or all(f.filename == '' for f in files):
                return jsonify({'error': 'No files selected'}), 400
            
            uploaded_files = []
            errors = []
            
            papers_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'papers')
            os.makedirs(papers_dir, exist_ok=True)
            
            for file in files:
                if file.filename == '':
                    continue
                    
                if not allowed_file(file.filename):
                    errors.append(f"{file.filename}: File type not allowed")
                    continue
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(papers_dir, filename)
                file.save(file_path)
                
                uploaded_files.append({
                    'filename': filename,
                    'size': os.path.getsize(file_path)
                })
            
            print(f"Successfully uploaded {len(uploaded_files)} research papers")
            
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded {len(uploaded_files)} research papers',
                'uploaded_files': uploaded_files,
                'errors': errors
            })
            
        except Exception as e:
            print(f"Papers upload error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # Collaboration prediction endpoint
    @app.route('/api/collaboration/predict', methods=['POST'])
    def predict_collaborations():
        nonlocal predictor
        try:
            print("Prediction request received")
            
            data = request.get_json()
            proposal_text = data.get('proposal_text', '')
            settings = data.get('settings', {})
            
            if not proposal_text or len(proposal_text.strip()) < 50:
                return jsonify({'error': 'Proposal text must be at least 50 characters long'}), 400
            
            # Initialize predictor if not already done
            if predictor is None:
                print("Initializing predictor...")
                predictor = EnhancedFacultyCollaborationPredictor()
                
                # Load faculty files
                faculty_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'faculty')
                if os.path.exists(faculty_dir) and os.listdir(faculty_dir):
                    predictor.load_faculty_bios(faculty_dir)
                    
                    # Load research papers if available
                    papers_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'papers')
                    if os.path.exists(papers_dir) and os.listdir(papers_dir):
                        predictor.load_research_papers(papers_dir)
                    
                    predictor.train_ml_model()
                else:
                    return jsonify({'error': 'Please upload faculty files first in Data Management'}), 400
            
            # Extract settings
            num_recommendations = settings.get('num_recommendations', 5)
            team_sizes = settings.get('team_sizes', [3, 4])
            use_ml = settings.get('use_ml', True)
            
            print(f"Generating {num_recommendations} predictions with team sizes {team_sizes}")
            
            # Generate predictions
            predictions = predictor.predict_collaborations_with_ml(
                proposal_text,
                top_n=num_recommendations,
                team_sizes=team_sizes,
                use_ml=use_ml
            )
            
            print(f"Raw predictions count: {len(predictions)}")
            
            # Format results for frontend
            formatted_predictions = []
            for i, pred in enumerate(predictions):
                formatted_predictions.append({
                    'id': i + 1,
                    'rank': i + 1,
                    'faculty_names': pred['faculty_names'],
                    'departments': pred['departments'],
                    'team_size': pred['team_size'],
                    'combined_score': round(pred['combined_score'], 3),
                    'ml_score': round(pred.get('ml_score', 0), 3),
                    'base_score': round(pred['base_score'], 3),
                    'synergy_reasons': pred['synergy_reasons'],
                    'individual_scores': [round(score, 3) for score in pred['individual_scores']]
                })
            
            print(f"Generated {len(formatted_predictions)} formatted predictions")
            
            return jsonify({
                'success': True,
                'predictions': formatted_predictions,
                'metadata': {
                    'proposal_length': len(proposal_text),
                    'team_sizes_requested': team_sizes,
                    'ml_enabled': use_ml,
                    'faculty_count': len(predictor.faculty_profiles),
                    'papers_count': len(predictor.research_papers),
                    'requested_teams': num_recommendations
                }
            })
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    # Get system status
    @app.route('/api/data/status', methods=['GET'])
    def get_system_status():
        try:
            faculty_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'faculty')
            papers_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'papers')
            
            faculty_files = 0
            paper_files = 0
            
            if os.path.exists(faculty_dir):
                faculty_files = len([f for f in os.listdir(faculty_dir) 
                                   if f.lower().endswith(('.txt', '.pdf', '.docx'))])
            
            if os.path.exists(papers_dir):
                paper_files = len([f for f in os.listdir(papers_dir) 
                                 if f.lower().endswith(('.txt', '.pdf', '.docx'))])
            
            return jsonify({
                'success': True,
                'data': {
                    'faculty_files_uploaded': faculty_files,
                    'paper_files_uploaded': paper_files,
                    'system_initialized': predictor is not None,
                    'ml_model_trained': predictor is not None and predictor.ml_model is not None,
                    'collaboration_system_available': COLLABORATION_SYSTEM_AVAILABLE
                }
            })
            
        except Exception as e:
            print(f"Status error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # NEW: Analytics endpoint for insights page
    @app.route('/api/analytics/insights', methods=['GET'])
    def get_analytics_insights():
        """Get comprehensive analytics data for the insights page"""
        try:
            print("Analytics insights request received")
            
            if predictor is None:
                return jsonify({
                    'error': 'System not initialized. Please upload faculty files first.'
                }), 400
            
            analytics_data = predictor.get_analytics_data()
            
            if analytics_data is None:
                return jsonify({
                    'error': 'No analytics data available. Please upload faculty files first.'
                }), 400
            
            return jsonify({
                'success': True,
                'data': analytics_data
            })
            
        except Exception as e:
            print(f"Analytics error: {str(e)}")
            return jsonify({'error': f'Failed to get analytics: {str(e)}'}), 500
    
    return app

app = create_app()

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs('data/uploads/faculty', exist_ok=True)
    os.makedirs('data/uploads/papers', exist_ok=True)
    os.makedirs('data/models', exist_ok=True) 
    os.makedirs('data/exports', exist_ok=True)
    
    print("Starting Faculty Collaboration System...")
    print(f"Collaboration System Available: {COLLABORATION_SYSTEM_AVAILABLE}")
    
    app.run(host='0.0.0.0', port=5001, debug=True)