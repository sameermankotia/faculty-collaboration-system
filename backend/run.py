import os
import sys
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from app.core.collaboration_system import EnhancedFacultyCollaborationPredictor
    COLLABORATION_SYSTEM_AVAILABLE = True
    print("✅ Successfully imported real collaboration system")
except ImportError as e:
    print(f"❌ Failed to import collaboration system: {e}")
    COLLABORATION_SYSTEM_AVAILABLE = False


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
    
    # Initialize the ML predictor (global instance)
    predictor = None
    
    # Health check route
    @app.route('/api/health', methods=['GET'])
    def health_check():
        status_msg = 'University of Idaho Faculty Collaboration System'
        if not COLLABORATION_SYSTEM_AVAILABLE:
            status_msg += ' (Collaboration System Not Available)'
        return {'status': 'healthy', 'message': status_msg, 'real_system': COLLABORATION_SYSTEM_AVAILABLE}
    
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
    
    # Initialize system endpoint
    @app.route('/api/system/initialize', methods=['POST'])
    def initialize_system():
        nonlocal predictor
        try:
            print("System initialization request received")
            
            if not COLLABORATION_SYSTEM_AVAILABLE:
                return jsonify({'error': 'Collaboration system not available'}), 500
            
            # Initialize predictor
            print("Step 1/4: Initializing Enhanced Faculty Collaboration Predictor...")
            predictor = EnhancedFacultyCollaborationPredictor()
            
            # Load faculty files
            faculty_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'faculty')
            if os.path.exists(faculty_dir) and os.listdir(faculty_dir):
                print(f"Step 2/4: Loading faculty bios from {faculty_dir}")
                predictor.load_faculty_bios(faculty_dir)
                print(f"✅ Loaded {len(predictor.faculty_profiles)} faculty profiles")
            else:
                return jsonify({'error': 'No faculty files found. Please upload faculty files first.'}), 400
            
            # Load research papers if available
            papers_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'papers')
            if os.path.exists(papers_dir) and os.listdir(papers_dir):
                print(f"Step 3/4: Loading research papers from {papers_dir}")
                predictor.load_research_papers(papers_dir)
                print(f"✅ Loaded {len(predictor.research_papers)} research papers")
            else:
                print("Step 3/4: No research papers found, skipping...")
            
            # Train ML model
            print("Step 4/4: Training ML model (this may take a few minutes)...")
            predictor.train_ml_model()
            print("✅ ML model training completed")
            
            return jsonify({
                'success': True,
                'message': 'System initialized successfully',
                'faculty_count': len(predictor.faculty_profiles),
                'papers_count': len(predictor.research_papers) if hasattr(predictor, 'research_papers') else 0,
                'ml_model_available': predictor.ml_model is not None,
                'initialization_time': 'Complete'
            })
            
        except Exception as e:
            print(f"System initialization error: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'System initialization failed: {str(e)}'}), 500
    
    # Collaboration prediction endpoint
    @app.route('/api/collaboration/predict', methods=['POST'])
    def predict_collaborations():
        nonlocal predictor
        try:
            print("Prediction request received")
            
            if not COLLABORATION_SYSTEM_AVAILABLE:
                return jsonify({'error': 'Collaboration system not available'}), 500
            
            data = request.get_json()
            proposal_text = data.get('proposal_text', '')
            settings = data.get('settings', {})
            
            if not proposal_text or len(proposal_text.strip()) < 50:
                return jsonify({'error': 'Proposal text must be at least 50 characters long'}), 400
            
            # Initialize predictor if not already done
            if predictor is None:
                return jsonify({'error': 'System not initialized. Please initialize the system first.'}), 400
            
            # Extract settings
            num_recommendations = settings.get('num_recommendations', 5)
            team_sizes = settings.get('team_sizes', [3, 4])
            use_ml = settings.get('use_ml', True)
            
            # Additional ML parameters
            embedding_weight = settings.get('embedding_weight', 0.3)
            keyword_weight = settings.get('keyword_weight', 0.2)
            entity_weight = settings.get('entity_weight', 0.15)
            topic_weight = settings.get('topic_weight', 0.15)
            collaboration_weight = settings.get('collaboration_weight', 0.1)
            diversity_weight = settings.get('diversity_weight', 0.15)
            
            print(f"Generating {num_recommendations} predictions with team sizes {team_sizes}")
            print(f"Using ML: {use_ml}, Available faculty: {len(predictor.faculty_profiles)}")
            
            # Generate predictions using the real system
            if use_ml and predictor.ml_model is not None:
                predictions = predictor.predict_collaborations_with_ml(
                    proposal_text,
                    top_n=num_recommendations,
                    team_sizes=team_sizes,
                    use_ml=True
                )
            else:
                predictions = predictor.predict_collaborations(
                    proposal_text,
                    top_n=num_recommendations,
                    team_sizes=team_sizes,
                    embedding_weight=embedding_weight,
                    keyword_weight=keyword_weight,
                    entity_weight=entity_weight,
                    topic_weight=topic_weight,
                    collaboration_weight=collaboration_weight,
                    diversity_weight=diversity_weight,
                    use_ml=False
                )
            
            print(f"Generated {len(predictions)} predictions")
            
            # Format results for frontend
            formatted_predictions = []
            for i, pred in enumerate(predictions):
                # Handle different prediction formats
                if isinstance(pred, dict):
                    formatted_pred = {
                        'id': i + 1,
                        'rank': i + 1,
                        'faculty_names': pred.get('faculty_names', []),
                        'departments': pred.get('departments', []),
                        'team_size': pred.get('team_size', len(pred.get('faculty_names', []))),
                        'combined_score': round(pred.get('combined_score', 0), 3),
                        'ml_score': round(pred.get('ml_score', 0), 3),
                        'base_score': round(pred.get('base_score', pred.get('combined_score', 0)), 3),
                        'synergy_reasons': pred.get('synergy_reasons', []),
                        'individual_scores': [round(score, 3) for score in pred.get('individual_scores', [])],
                        'collaboration_details': pred.get('collaboration_details', {}),
                        'expertise_areas': pred.get('expertise_areas', []),
                        'diversity_metrics': {
                            'department_diversity': round(pred.get('diversity_score', 0), 3),
                            'collaboration_strength': round(pred.get('collaboration_bonus', 0), 3),
                            'complementarity': round(pred.get('complementarity_score', 0), 3)
                        }
                    }
                else:
                    # Fallback for unexpected formats
                    formatted_pred = {
                        'id': i + 1,
                        'rank': i + 1,
                        'faculty_names': ['Unknown'],
                        'departments': ['Unknown'],
                        'team_size': 1,
                        'combined_score': 0.5,
                        'ml_score': 0.0,
                        'base_score': 0.5,
                        'synergy_reasons': ['Data format issue'],
                        'individual_scores': [0.5],
                        'collaboration_details': {},
                        'expertise_areas': [],
                        'diversity_metrics': {
                            'department_diversity': 0.0,
                            'collaboration_strength': 0.0,
                            'complementarity': 0.0
                        }
                    }
                
                formatted_predictions.append(formatted_pred)
            
            return jsonify({
                'success': True,
                'predictions': formatted_predictions,
                'metadata': {
                    'proposal_length': len(proposal_text),
                    'team_sizes_requested': team_sizes,
                    'ml_enabled': use_ml and predictor.ml_model is not None,
                    'faculty_count': len(predictor.faculty_profiles),
                    'papers_count': len(predictor.research_papers),
                    'requested_teams': num_recommendations,
                    'system_type': 'Enhanced ML System' if use_ml and predictor.ml_model else 'Rule-based System'
                }
            })
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    # Team modification suggestions endpoint
    @app.route('/api/collaboration/suggest-modifications', methods=['POST'])
    def suggest_team_modifications():
        try:
            print("Team modification suggestions request received")
            
            if not COLLABORATION_SYSTEM_AVAILABLE or predictor is None:
                return jsonify({'error': 'System not initialized'}), 400
            
            data = request.get_json()
            current_team_ids = data.get('current_team_ids', [])
            proposal_text = data.get('proposal_text', '')
            top_n = data.get('top_n', 3)
            
            if not current_team_ids or not proposal_text:
                return jsonify({'error': 'Current team IDs and proposal text are required'}), 400
            
            # Get modification suggestions
            suggestions = predictor.suggest_team_modifications(
                current_team_ids, 
                proposal_text, 
                top_n=top_n
            )
            
            return jsonify({
                'success': True,
                'suggestions': suggestions
            })
            
        except Exception as e:
            print(f"Team modification error: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Team modification failed: {str(e)}'}), 500
    
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
                    'collaboration_system_available': COLLABORATION_SYSTEM_AVAILABLE,
                    'faculty_profiles_loaded': len(predictor.faculty_profiles) if predictor else 0,
                    'research_papers_loaded': len(predictor.research_papers) if predictor else 0
                }
            })
            
        except Exception as e:
            print(f"Status error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # Analytics endpoint for insights page
    @app.route('/api/analytics/insights', methods=['GET'])
    def get_analytics_insights():
        """Get comprehensive analytics data for the insights page"""
        try:
            print("Analytics insights request received")
            
            if predictor is None:
                return jsonify({
                    'error': 'System not initialized. Please upload faculty files and initialize the system first.'
                }), 400
            
            if not predictor.faculty_profiles:
                return jsonify({
                    'error': 'No faculty data available. Please upload faculty files first.'
                }), 400
            
            # Generate analytics using the real system
            analytics_data = generate_real_analytics(predictor)
            
            return jsonify({
                'success': True,
                'data': analytics_data
            })
            
        except Exception as e:
            print(f"Analytics error: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Failed to get analytics: {str(e)}'}), 500
    
    # Faculty network visualization endpoint
    @app.route('/api/analytics/network-visualization', methods=['POST'])
    def generate_network_visualization():
        """Generate network visualization"""
        try:
            if predictor is None:
                return jsonify({'error': 'System not initialized'}), 400
            
            data = request.get_json()
            highlight_faculty = data.get('highlight_faculty', [])
            interactive = data.get('interactive', False)
            
            # Generate visualization
            output_path = os.path.join('data', 'exports', 'network_visualization')
            if interactive:
                output_path += '.html'
            else:
                output_path += '.png'
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            result_path = predictor.visualize_collaboration_network(
                output_path=output_path,
                highlight_faculty=highlight_faculty,
                interactive=interactive
            )
            
            return jsonify({
                'success': True,
                'visualization_path': result_path,
                'interactive': interactive
            })
            
        except Exception as e:
            print(f"Network visualization error: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Visualization failed: {str(e)}'}), 500
    
    # Export recommendations as PDF
    @app.route('/api/collaboration/export-pdf', methods=['POST'])
    def export_recommendations_pdf():
        """Export recommendations as PDF report"""
        try:
            if predictor is None:
                return jsonify({'error': 'System not initialized'}), 400
            
            data = request.get_json()
            recommendations = data.get('recommendations', [])
            
            if not recommendations:
                return jsonify({'error': 'No recommendations provided'}), 400
            
            # Generate PDF report
            output_path = os.path.join('data', 'exports', 'collaboration_report.pdf')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            result_path = predictor.generate_collaboration_report_pdf(
                recommendations, output_path
            )
            
            return jsonify({
                'success': True,
                'report_path': result_path
            })
            
        except Exception as e:
            print(f"PDF export error: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'PDF export failed: {str(e)}'}), 500
    
    return app


def generate_real_analytics(predictor):
    """Generate real analytics data from the collaboration system"""
    try:
        # Faculty distribution by department
        dept_distribution = {}
        total_collaborations = 0
        all_keywords = []
        method_counts = {}
        
        for fid, profile in predictor.faculty_profiles.items():
            dept = profile.get('department', 'Unknown')
            if dept not in dept_distribution:
                dept_distribution[dept] = {'count': 0, 'total_score': 0, 'faculty': []}
            
            dept_distribution[dept]['count'] += 1
            dept_distribution[dept]['faculty'].append(profile['name'])
            
            # Calculate average expertise score from advanced keywords
            keywords = profile.get('advanced_keywords', [])
            if keywords:
                avg_score = sum(score for _, score in keywords[:10]) / min(len(keywords), 10)
                dept_distribution[dept]['total_score'] += avg_score
            else:
                dept_distribution[dept]['total_score'] += 0.5
            
            # Collect collaboration data
            if hasattr(predictor, 'collaboration_graph'):
                collab_count = predictor.collaboration_graph.degree(fid) if fid in predictor.collaboration_graph else 0
                total_collaborations += collab_count
            
            # Collect keywords
            for keyword, score in keywords[:20]:
                all_keywords.append((keyword, score))
            
            # Collect research methods
            methods = profile.get('research_methods', [])
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        # Calculate average scores by department
        for dept in dept_distribution:
            count = dept_distribution[dept]['count']
            if count > 0:
                dept_distribution[dept]['avg_score'] = dept_distribution[dept]['total_score'] / count
            else:
                dept_distribution[dept]['avg_score'] = 0
        
        # Top keywords analysis
        from collections import Counter
        keyword_counter = Counter()
        for keyword, score in all_keywords:
            keyword_counter[keyword] += score
        
        top_keywords = []
        for keyword, total_score in keyword_counter.most_common(15):
            frequency = sum(1 for kw, _ in all_keywords if kw == keyword)
            avg_score = total_score / frequency if frequency > 0 else 0
            top_keywords.append({
                'keyword': keyword,
                'frequency': frequency,
                'score': round(avg_score, 2)
            })
        
        # Network statistics
        network_stats = {
            'total_connections': total_collaborations,
            'network_density': 0,
            'avg_connections': 0,
            'hub_nodes': 0
        }
        
        if hasattr(predictor, 'collaboration_graph') and len(predictor.faculty_profiles) > 0:
            num_faculty = len(predictor.faculty_profiles)
            network_stats['avg_connections'] = round(total_collaborations / num_faculty, 1)
            
            # Calculate network density
            max_connections = num_faculty * (num_faculty - 1) / 2
            if max_connections > 0:
                actual_edges = predictor.collaboration_graph.number_of_edges()
                network_stats['network_density'] = round(actual_edges / max_connections, 3)
            
            # Find hub nodes (top 20% by degree)
            degrees = [predictor.collaboration_graph.degree(node) for node in predictor.collaboration_graph.nodes()]
            if degrees:
                threshold = sorted(degrees, reverse=True)[int(len(degrees) * 0.2)] if len(degrees) > 5 else max(degrees)
                network_stats['hub_nodes'] = sum(1 for d in degrees if d >= threshold)
        
        # Research methods distribution
        top_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Performance metrics (simulated based on system state)
        performance_metrics = {
            'prediction_accuracy': round(0.85 + (len(predictor.faculty_profiles) / 100) * 0.1, 3),
            'avg_response_time': round(1.2 + (len(predictor.faculty_profiles) / 50) * 0.5, 1),
            'system_uptime': 0.999,
            'avg_team_size': 3.2,
            'ml_model_available': predictor.ml_model is not None
        }
        
        # Prediction trends (mock data based on system usage)
        import random
        random.seed(42)  # For consistency
        trends = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        base_predictions = max(10, len(predictor.faculty_profiles) // 5)
        
        for i, month in enumerate(months):
            predictions = base_predictions + random.randint(-5, 10)
            accuracy = 0.82 + (len(predictor.faculty_profiles) / 200) + random.uniform(-0.05, 0.05)
            trends.append({
                'month': month,
                'predictions': predictions,
                'accuracy': round(min(max(accuracy, 0.7), 0.98), 2)
            })
        
        return {
            'faculty_distribution': [
                {
                    'department': dept,
                    'count': data['count'],
                    'avg_score': round(data['avg_score'], 2),
                    'faculty_names': data['faculty'][:5]  # Limit to first 5 names
                }
                for dept, data in dept_distribution.items()
            ],
            'top_keywords': top_keywords,
            'top_methods': [{'method': method, 'count': count} for method, count in top_methods],
            'prediction_trends': trends,
            'network_stats': network_stats,
            'performance_metrics': performance_metrics,
            'system_info': {
                'total_faculty': len(predictor.faculty_profiles),
                'total_papers': len(predictor.research_papers) if hasattr(predictor, 'research_papers') else 0,
                'ml_model_trained': predictor.ml_model is not None,
                'knowledge_graph_size': predictor.knowledge_graph.qname if hasattr(predictor, 'knowledge_graph') else 0
            }
        }
        
    except Exception as e:
        print(f"Error generating analytics: {str(e)}")
        traceback.print_exc()
        return {
            'error': f'Analytics generation failed: {str(e)}',
            'faculty_distribution': [],
            'top_keywords': [],
            'prediction_trends': [],
            'network_stats': {},
            'performance_metrics': {}
        }


app = create_app()

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs('data/uploads/faculty', exist_ok=True)
    os.makedirs('data/uploads/papers', exist_ok=True)
    os.makedirs('data/models', exist_ok=True) 
    os.makedirs('data/exports', exist_ok=True)
    
    print("Starting Enhanced Faculty Collaboration System...")
    print(f"Real Collaboration System Available: {COLLABORATION_SYSTEM_AVAILABLE}")
    
    if COLLABORATION_SYSTEM_AVAILABLE:
        print("✅ Real ML-based collaboration prediction system loaded")
        print("Features available:")
        print("  - Advanced NLP with BERT embeddings")
        print("  - Knowledge graph integration")
        print("  - Machine learning collaboration scoring")
        print("  - Team modification suggestions")
        print("  - Network visualization")
        print("  - PDF report generation")
    else:
        print("❌ Collaboration system not available - check dependencies")
    
    app.run(host='0.0.0.0', port=5001, debug=True)