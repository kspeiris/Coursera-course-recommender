import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import time
import os
import random
from functools import wraps
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'edu-recommend-secret-key-2024'

# Theme configurations (same as before)
THEMES = {
    'default': {'name': 'Ocean Blue', 'primary': '#6366f1', 'secondary': '#10b981', 'accent': '#f59e0b', 'background': '#0f172a', 'surface': '#1e293b'},
    'purple': {'name': 'Royal Purple', 'primary': '#8b5cf6', 'secondary': '#ec4899', 'accent': '#06b6d4', 'background': '#1e1b4b', 'surface': '#312e81'},
    'green': {'name': 'Emerald Forest', 'primary': '#059669', 'secondary': '#dc2626', 'accent': '#d97706', 'background': '#052e16', 'surface': '#065f46'},
    'orange': {'name': 'Sunset Orange', 'primary': '#ea580c', 'secondary': '#db2777', 'accent': '#ca8a04', 'background': '#451a03', 'surface': '#9a3412'},
    'pink': {'name': 'Cotton Candy', 'primary': '#db2777', 'secondary': '#7c3aed', 'accent': '#0891b2', 'background': '#500724', 'surface': '#831843'},
    'light': {'name': 'Light Mode', 'primary': '#3b82f6', 'secondary': '#10b981', 'accent': '#f59e0b', 'background': '#ffffff', 'surface': '#f8fafc'}
}

# Enhanced sample data with realistic ratings and reviews
def generate_realistic_ratings():
    """Generate realistic rating distributions for courses"""
    return {
        'rating': round(random.uniform(3.8, 4.9), 1),
        'reviews': random.randint(500, 50000),
        'stars': {
            '5': random.randint(40, 85),
            '4': random.randint(15, 40),
            '3': random.randint(5, 20),
            '2': random.randint(1, 10),
            '1': random.randint(0, 5)
        }
    }

# Load dataset with enhanced rating system
try:
    data = pd.read_csv('Coursera_courses_catalog.csv')
    logger.info("Dataset loaded successfully")
    
    required_columns = ['course_name', 'category', 'course_skills', 'course_link', 'time_required']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
            
    data = data.dropna(subset=['course_name', 'course_link'])
    data['category'] = data['category'].fillna('General')
    data['course_skills'] = data['course_skills'].fillna('[]')
    
    # Add rating columns if they don't exist
    if 'rating' not in data.columns:
        data['rating'] = [generate_realistic_ratings()['rating'] for _ in range(len(data))]
    if 'reviews' not in data.columns:
        data['reviews'] = [generate_realistic_ratings()['reviews'] for _ in range(len(data))]
    if 'enrollment' not in data.columns:
        data['enrollment'] = [random.randint(10000, 200000) for _ in range(len(data))]
        
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    # Create comprehensive sample data with realistic ratings
    sample_courses = []
    course_templates = [
        {
            'name': 'Machine Learning Specialization',
            'category': 'Data Science',
            'skills': ['Python', 'Machine Learning', 'Deep Learning', 'Statistics', 'TensorFlow'],
            'link': 'https://coursera.org/specializations/machine-learning',
            'time': '4-6 hours',
            'difficulty': 'Intermediate'
        },
        {
            'name': 'Python for Everybody',
            'category': 'Computer Science',
            'skills': ['Python', 'Programming', 'Web Development', 'Data Structures'],
            'link': 'https://coursera.org/specializations/python',
            'time': '3-5 hours',
            'difficulty': 'Beginner'
        },
        {
            'name': 'Data Science Fundamentals',
            'category': 'Data Science',
            'skills': ['Data Analysis', 'Statistics', 'Python', 'SQL', 'Pandas'],
            'link': 'https://coursera.org/specializations/data-science',
            'time': '5-7 hours',
            'difficulty': 'Intermediate'
        },
        {
            'name': 'Business Analytics',
            'category': 'Business',
            'skills': ['Excel', 'Data Analysis', 'Statistics', 'Business Intelligence'],
            'link': 'https://coursera.org/specializations/business-analytics',
            'time': '4-5 hours',
            'difficulty': 'Beginner'
        },
        {
            'name': 'AI For Everyone',
            'category': 'Artificial Intelligence',
            'skills': ['AI', 'Machine Learning', 'Business Strategy', 'Ethics'],
            'link': 'https://coursera.org/learn/ai-for-everyone',
            'time': '2-3 hours',
            'difficulty': 'Beginner'
        },
        {
            'name': 'Web Development Bootcamp',
            'category': 'Computer Science',
            'skills': ['HTML', 'CSS', 'JavaScript', 'React', 'Node.js'],
            'link': 'https://coursera.org/specializations/web-development',
            'time': '6-8 hours',
            'difficulty': 'Intermediate'
        },
        {
            'name': 'Digital Marketing Specialization',
            'category': 'Marketing',
            'skills': ['Marketing', 'SEO', 'Social Media', 'Analytics', 'Strategy'],
            'link': 'https://coursera.org/specializations/digital-marketing',
            'time': '3-4 hours',
            'difficulty': 'Beginner'
        },
        {
            'name': 'Deep Learning Specialization',
            'category': 'Artificial Intelligence',
            'skills': ['Deep Learning', 'Neural Networks', 'Python', 'TensorFlow', 'Keras'],
            'link': 'https://coursera.org/specializations/deep-learning',
            'time': '5-7 hours',
            'difficulty': 'Advanced'
        },
        {
            'name': 'Cloud Computing Fundamentals',
            'category': 'Computer Science',
            'skills': ['AWS', 'Cloud Computing', 'DevOps', 'Infrastructure'],
            'link': 'https://coursera.org/specializations/cloud-computing',
            'time': '4-6 hours',
            'difficulty': 'Intermediate'
        },
        {
            'name': 'UX Design Principles',
            'category': 'Design',
            'skills': ['UX Design', 'UI Design', 'User Research', 'Prototyping'],
            'link': 'https://coursera.org/specializations/ux-design',
            'time': '3-5 hours',
            'difficulty': 'Beginner'
        }
    ]
    
    for template in course_templates:
        ratings_data = generate_realistic_ratings()
        sample_courses.append({
            'course_name': template['name'],
            'category': template['category'],
            'course_skills': template['skills'],
            'course_link': template['link'],
            'time_required': template['time'],
            'difficulty': template['difficulty'],
            'rating': ratings_data['rating'],
            'reviews': ratings_data['reviews'],
            'enrollment': random.randint(10000, 200000),
            'stars_distribution': ratings_data['stars']
        })
    
    data = pd.DataFrame(sample_courses)

# Predefined options (same as before)
predefined_categories = ["Data Science", "Business", "Health", "Arts and Humanities", "Computer Science", "Social Sciences", "Engineering", "Mathematics", "Personal Development", "Language Learning", "Artificial Intelligence", "Marketing", "Finance"]
predefined_skills = ["Python", "Machine Learning", "Data Analysis", "AI", "Statistics", "Deep Learning", "Natural Language Processing", "Data Visualization", "SQL", "R Programming", "Excel", "Tableau", "Power BI", "Java", "JavaScript", "React", "Node.js", "Cloud Computing", "AWS", "HTML", "CSS", "Marketing", "Finance", "Leadership"]
predefined_difficulties = ["Beginner", "Intermediate", "Advanced", "Mixed"]
predefined_languages = ["English", "Spanish", "French", "German", "Chinese", "Portuguese", "Italian", "Japanese"]
predefined_subtitles = ["English", "Spanish", "French", "Portuguese", "German", "Chinese", "Arabic"]
predefined_durations = ["1-3 hours", "4-6 hours", "7-10 hours", "11-15 hours", "16+ hours"]

# User rating storage (in production, use a database)
user_ratings = {}

# Helper functions
def add_loading_animation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        time.sleep(0.5)
        return func(*args, **kwargs)
    return wrapper

def convert_time_to_hours(time_str):
    if pd.isna(time_str):
        return (0, 0)
    time_str = str(time_str).lower()
    match_range = re.match(r"(\d+)-(\d+)\s*hours?", time_str)
    if match_range:
        return (int(match_range.group(1)), int(match_range.group(2)))
    match_single = re.match(r"(\d+)\s*hours?", time_str)
    if match_single:
        value = int(match_single.group(1))
        return (value, value)
    match_weeks = re.match(r"(\d+)\s*weeks?", time_str)
    if match_weeks:
        weeks = int(match_weeks.group(1))
        return (weeks * 5, weeks * 10)
    return (0, 0)

def filter_courses_by_time(data, user_time):
    if not user_time or user_time == "Any":
        return data
    min_time, max_time = convert_time_to_hours(user_time)
    if min_time == 0 and max_time == 0:
        return data
    def time_within_range(time_str):
        course_min_time, course_max_time = convert_time_to_hours(time_str)
        return (course_max_time >= min_time) and (course_min_time <= max_time)
    return data[data['time_required'].apply(time_within_range)]

@add_loading_animation
def filter_courses_by_preferences(data, user_input):
    user_topic, user_skills, user_category, user_difficulty, user_language, user_time, user_subtitles = user_input
    filtered_data = data.copy()
    if user_category and user_category != "Any":
        filtered_data = filtered_data[filtered_data['category'].str.contains(user_category, case=False, na=False)]
    filtered_data = filter_courses_by_time(filtered_data, user_time)
    if user_skills and user_skills != "Any":
        filtered_data = filtered_data[
            filtered_data['course_skills'].apply(
                lambda x: user_skills.lower() in str(x).lower() if pd.notna(x) else False
            )
        ]
    if user_difficulty and user_difficulty != "Any" and 'difficulty' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['difficulty'].str.contains(user_difficulty, case=False, na=False)]
    return filtered_data

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))

@add_loading_animation
def recommend_courses(user_input, filtered_courses, top_n=8):
    if filtered_courses.empty:
        return []
    
    user_topic, user_skills, user_category, user_difficulty, user_language, user_time, user_subtitles = user_input
    user_query = f"{user_topic} {user_skills} {user_category}"
    
    try:
        filtered_courses = filtered_courses.copy()
        filtered_courses['course_features'] = (
            filtered_courses['course_name'] + " " + 
            filtered_courses['category'] + " " +
            filtered_courses['course_skills'].apply(
                lambda x: ' '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else str(x)
            )
        )
        
        filtered_courses = filtered_courses[filtered_courses['course_features'].str.strip().astype(bool)]
        
        if filtered_courses.empty:
            return []
        
        tfidf_matrix = vectorizer.fit_transform(filtered_courses['course_features'])
        user_input_tfidf = vectorizer.transform([user_query])
        
        cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)
        
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        top_courses = []
        for i in sim_scores[:top_n]:
            course = filtered_courses.iloc[i[0]]
            
            # Generate star distribution if not present
            if 'stars_distribution' not in course or not course['stars_distribution']:
                stars = generate_realistic_ratings()['stars']
            else:
                stars = course['stars_distribution']
            
            # Calculate rating percentage
            total_reviews = sum(stars.values())
            rating_percentages = {str(k): round((v / total_reviews) * 100, 1) for k, v in stars.items()}
            
            course_data = {
                'name': course['course_name'],
                'link': course['course_link'],
                'category': course.get('category', 'General'),
                'skills': course.get('course_skills', '[]'),
                'duration': course.get('time_required', 'Not specified'),
                'difficulty': course.get('difficulty', 'Not specified'),
                'rating': course.get('rating', 4.5),
                'reviews': course.get('reviews', 0),
                'enrollment': course.get('enrollment', 0),
                'stars_distribution': stars,
                'rating_percentages': rating_percentages,
                'total_reviews': total_reviews,
                'similarity_score': min(round(i[1] * 100, 1), 99.9),
                'course_id': f"course_{len(top_courses) + 1}"
            }
            top_courses.append(course_data)
            
        return top_courses
        
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        return []

# Routes
@app.route('/')
def home():
    theme = request.args.get('theme', 'default')
    mode = request.args.get('mode', 'dark')
    
    # Store theme preferences in session
    session['theme'] = theme
    session['mode'] = mode
    
    return render_template('index.html', 
                         categories=predefined_categories,
                         skills=predefined_skills,
                         difficulties=predefined_difficulties,
                         languages=predefined_languages,
                         subtitles=predefined_subtitles,
                         durations=predefined_durations,
                         themes=THEMES,
                         current_theme=theme,
                         current_mode=mode)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_topic = request.form.get('topic', '').strip()
        user_skills = request.form.get('skills', 'Any')
        user_category = request.form.get('category', 'Any')
        user_difficulty = request.form.get('difficulty', 'Any')
        user_language = request.form.get('language', 'English')
        user_time = request.form.get('time', 'Any')
        user_subtitles = request.form.get('subtitles', 'English')
        theme = request.form.get('theme', 'default')
        mode = request.form.get('mode', 'dark')
        
        # Store theme preferences in session
        session['theme'] = theme
        session['mode'] = mode
        
        if not user_topic:
            return render_template('index.html', 
                                 categories=predefined_categories,
                                 skills=predefined_skills,
                                 difficulties=predefined_difficulties,
                                 languages=predefined_languages,
                                 subtitles=predefined_subtitles,
                                 durations=predefined_durations,
                                 themes=THEMES,
                                 current_theme=theme,
                                 current_mode=mode,
                                 error="Please enter a course topic")
        
        user_input = (user_topic, user_skills, user_category, user_difficulty, user_language, user_time, user_subtitles)
        
        filtered_courses = filter_courses_by_preferences(data, user_input)
        recommended_courses = recommend_courses(user_input, filtered_courses)
        
        session['last_search'] = {
            'query': user_topic,
            'results_count': len(recommended_courses),
            'timestamp': time.time()
        }
        
        return render_template('recommendations.html', 
                             recommended_courses=recommended_courses,
                             search_query=user_topic,
                             results_count=len(recommended_courses),
                             themes=THEMES,
                             current_theme=theme,
                             current_mode=mode,
                             filters_applied={
                                 'skills': user_skills,
                                 'category': user_category,
                                 'difficulty': user_difficulty,
                                 'duration': user_time
                             })
                             
    except Exception as e:
        logger.error(f"Error in recommendation route: {e}")
        return render_template('index.html',
                             categories=predefined_categories,
                             skills=predefined_skills,
                             difficulties=predefined_difficulties,
                             languages=predefined_languages,
                             subtitles=predefined_subtitles,
                             durations=predefined_durations,
                             themes=THEMES,
                             current_theme=request.form.get('theme', 'default'),
                             current_mode=request.form.get('mode', 'dark'),
                             error="An error occurred while processing your request")

# Theme API endpoints
@app.route('/api/theme/update', methods=['POST'])
def update_theme():
    try:
        data = request.get_json()
        theme = data.get('theme', 'default')
        mode = data.get('mode', 'dark')
        
        # Validate theme and mode
        if theme not in THEMES:
            theme = 'default'
        if mode not in ['light', 'dark']:
            mode = 'dark'
        
        # Store in session
        session['theme'] = theme
        session['mode'] = mode
            
        return jsonify({
            'success': True,
            'theme': theme,
            'mode': mode,
            'message': 'Theme preferences updated successfully'
        })
    except Exception as e:
        logger.error(f"Error updating theme: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/theme/preferences')
def get_theme_preferences():
    """Get current theme preferences from session"""
    try:
        theme = session.get('theme', 'default')
        mode = session.get('mode', 'dark')
        
        return jsonify({
            'success': True,
            'theme': theme,
            'mode': mode
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Rating API endpoints
@app.route('/api/rate_course', methods=['POST'])
def rate_course():
    try:
        rating_data = request.get_json()
        course_id = rating_data.get('course_id')
        rating = rating_data.get('rating')
        user_id = session.get('user_id', 'anonymous')
        
        if not course_id or rating not in [1, 2, 3, 4, 5]:
            return jsonify({'success': False, 'error': 'Invalid rating data'})
        
        # Store rating (in production, use database)
        if user_id not in user_ratings:
            user_ratings[user_id] = {}
        user_ratings[user_id][course_id] = rating
        
        # Log the rating
        logger.info(f"User {user_id} rated course {course_id}: {rating} stars")
        
        return jsonify({
            'success': True,
            'message': 'Rating submitted successfully',
            'user_rating': rating
        })
        
    except Exception as e:
        logger.error(f"Error submitting rating: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_rating/<course_id>')
def get_rating(course_id):
    try:
        user_id = session.get('user_id', 'anonymous')
        user_rating = user_ratings.get(user_id, {}).get(course_id)
        
        return jsonify({
            'success': True,
            'user_rating': user_rating
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/course_details/<course_name>')
def course_details(course_name):
    try:
        # Find course in dataset
        course = data[data['course_name'] == course_name].iloc[0]
        
        # Generate detailed rating information
        if 'stars_distribution' in course and course['stars_distribution']:
            stars = course['stars_distribution']
        else:
            stars = generate_realistic_ratings()['stars']
        
        total_reviews = sum(stars.values())
        rating_percentages = {str(k): round((v / total_reviews) * 100, 1) for k, v in stars.items()}
        
        course_details = {
            'name': course['course_name'],
            'rating': course.get('rating', 4.5),
            'reviews': course.get('reviews', total_reviews),
            'stars_distribution': stars,
            'rating_percentages': rating_percentages,
            'total_reviews': total_reviews,
            'enrollment': course.get('enrollment', 0),
            'duration': course.get('time_required', 'Not specified'),
            'difficulty': course.get('difficulty', 'Not specified'),
            'category': course.get('category', 'General')
        }
        
        return jsonify({'success': True, 'course': course_details})
        
    except Exception as e:
        logger.error(f"Error fetching course details: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'courses_count': len(data),
        'themes_available': list(THEMES.keys())
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found",
                         themes=THEMES,
                         current_theme=session.get('theme', 'default'),
                         current_mode=session.get('mode', 'dark')), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',
                         error_code=500,
                         error_message="Internal server error",
                         themes=THEMES,
                         current_theme=session.get('theme', 'default'),
                         current_mode=session.get('mode', 'dark')), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)