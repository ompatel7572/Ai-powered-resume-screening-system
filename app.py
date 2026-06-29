# Standard Library
import os
import json
from datetime import datetime
import time
import numpy as np
import pandas as pd


# Third-Party Libraries

import spacy
from spacy.matcher import PhraseMatcher
from spacy_layout import spaCyLayout
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, g, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    current_user,
    UserMixin
)
from sqlalchemy.exc import IntegrityError
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

# Own Modules
from Screening.rank import rank_resumes
from Parsing.parser import get_resume_text, append_to_json, parse_resume


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'your_secret_key'  # Required for session storage
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    # Added Gemini API key column
    llm_api_key = db.Column(db.String(200), nullable=True)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        raw_password = request.form.get('password', '')
        llm_api_key = request.form.get(
            'llm_api_key', '').strip()  # Get Gemini API key

        if not name or not email or not raw_password:
            session['message'] = 'All fields are required.'
            session['message_type'] = 'warning'
            return redirect(url_for('index'))

        try:
            hashed_password = bcrypt.generate_password_hash(
                raw_password).decode('utf-8')
            user = User(name=name, email=email, password=hashed_password,
                        llm_api_key=llm_api_key)
            db.session.add(user)
            db.session.commit()
            session['message'] = 'Account created successfully! Please login.'
            session['message_type'] = 'success'
            return redirect(url_for('index'))

        except IntegrityError:
            db.session.rollback()
            session['message'] = 'Email is already registered. Try logging in or use a different email.'
            session['message_type'] = 'danger'
            return redirect(url_for('index'))

        except Exception as e:
            db.session.rollback()
            session['message'] = 'An unexpected error occurred. Please try again later.'
            session['message_type'] = 'danger'
            print(f"Error during signup: {e}")
            return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        raw_password = request.form.get('password', '')

        if not email or not raw_password:
            session['message'] = 'Both email and password are required.'
            session['message_type'] = 'warning'
            return redirect(url_for('index'))

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, raw_password):
            login_user(user)
            session['name'] = user.name
            # session['message'] = 'Logged in successfully.'
            # session['message_type'] = 'success'
            return redirect(url_for('dashboard'))
        else:
            session['message'] = 'Login failed. Check email and password.'
            session['message_type'] = 'danger'
            return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Add these routes to your existing Flask application

@app.route('/settings')
@login_required
def settings():
    """Display user settings page."""
    # Load the JSON data
    user_identifier = str(current_user.id)
    json_file = f"parsed_data_{user_identifier}.json"
    json_data = "{}"  # Default empty data
    # This is just an example, adjust as per your model
    current_api_key = current_user.llm_api_key
    # genai.configure(api_key=current_api_key)

    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding="utf-8") as f:
                parsed_data = json.load(f)
                json_data = json.dumps(parsed_data, indent=2)
        except json.JSONDecodeError:
            json_data = "{}"  # Reset to empty if corrupted

    return render_template('settings.html', json_data=json_data, current_api_key=current_api_key)


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile information."""
    name = request.form.get('name', '').strip()

    if not name:
        session['message'] = 'Name cannot be empty.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    try:
        current_user.name = name
        db.session.commit()
        session['name'] = name  # Update session name
        session['message'] = 'Profile updated successfully.'
        session['message_type'] = 'success'
    except Exception as e:
        db.session.rollback()
        session['message'] = f'Error updating profile: {str(e)}'
        session['message_type'] = 'danger'

    return redirect(url_for('settings'))


@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    """Change user password."""
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # Validate inputs
    if not current_password or not new_password or not confirm_password:
        session['message'] = 'All password fields are required.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    if new_password != confirm_password:
        session['message'] = 'New passwords do not match.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    # Check current password
    if not bcrypt.check_password_hash(current_user.password, current_password):
        session['message'] = 'Current password is incorrect.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    try:
        # Update password
        hashed_password = bcrypt.generate_password_hash(
            new_password).decode('utf-8')
        current_user.password = hashed_password
        db.session.commit()

        session['message'] = 'Password changed successfully.'
        session['message_type'] = 'success'
    except Exception as e:
        db.session.rollback()
        session['message'] = f'Error changing password: {str(e)}'
        session['message_type'] = 'danger'

    return redirect(url_for('settings'))


@app.route('/change_api_key', methods=['POST'])
@login_required
def change_api_key():
    """Handle API key change."""
    current_api_key = request.form.get('current_api_key')
    new_api_key = request.form.get('new_api_key')
    confirm_api_key = request.form.get('confirm_api_key')

    # Validate the input
    if not current_api_key or not new_api_key or not confirm_api_key:
        session['message'] = 'All API key fields are required.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    if new_api_key != confirm_api_key:
        session['message'] = 'New API keys do not match.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    # Check if the current API key is correct (you need to implement this check)
    if current_api_key != current_user.llm_api_key:
        session['message'] = 'Current API key is incorrect.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    try:
        # Update the API key (you would store it in the user model)
        current_user.llm_api_key = new_api_key
        db.session.commit()

        session['message'] = 'API key changed successfully.'
        session['message_type'] = 'success'
    except Exception as e:
        db.session.rollback()
        session['message'] = f'Error changing API key: {str(e)}'
        session['message_type'] = 'danger'

    return redirect(url_for('settings'))


@app.route('/save_json_data', methods=['POST'])
@login_required
def save_json_data():
    """Save edited JSON data."""
    user_identifier = str(current_user.id)
    json_file = f"parsed_data_{user_identifier}.json"

    try:
        # Get JSON data from request
        data = request.json.get('data')
        parsed_json = json.loads(data)

        # Save to file
        with open(json_file, 'w', encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=2)

        # Also delete any ranking results since data has changed
        result_file = f"results_{user_identifier}.json"
        if os.path.exists(result_file):
            os.remove(result_file)

        return jsonify({
            'status': 'success',
            'message': 'Data saved successfully'
        })
    except json.JSONDecodeError:
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON format'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/dashboard')
@login_required
def dashboard():
    alert_message = session.pop('alert_message', None)
    # ranked_resumes = []

    # 🧠 Use user ID or email to create a unique file name
    user_identifier = str(current_user.id)

    json_file_1 = f"parsed_data_{user_identifier}.json"

    if os.path.exists(json_file_1) and os.path.getsize(json_file_1) > 0:
        try:
            with open(json_file_1, 'r', encoding="utf-8") as f:
                results = json.load(f)
                parsed_resumes = results
                # print(parsed_resumes)
        except json.JSONDecodeError:
            # print("Error: JSON file is corrupted!")
            parsed_resumes = []
    else:
        # print("Warning: JSON file is missing or empty!")
        parsed_resumes = []

    return render_template('dashboard.html',
                           parsed_resumes=parsed_resumes,
                           alert_message=alert_message,)


@app.route('/dashboard_candidate')
@login_required
def dashboard_candidate():

    user_identifier = str(current_user.id)

    json_file_1 = f"parsed_data_{user_identifier}.json"

    if os.path.exists(json_file_1) and os.path.getsize(json_file_1) > 0:
        try:
            with open(json_file_1, 'r', encoding="utf-8") as f:
                results = json.load(f)
                parsed_resumes = results
                # print(parsed_resumes)
        except json.JSONDecodeError:
            # print("Error: JSON file is corrupted!")
            parsed_resumes = []
    else:
        # print("Warning: JSON file is missing or empty!")
        parsed_resumes = []

    return render_template('dashboard_candidate.html',
                           parsed_resumes=parsed_resumes,
                           )


def load_resources():
    """Loads NLP model, layout processor, and skill extractor once."""

    def display_table(df: pd.DataFrame) -> str:
        lines = []

        # Header
        lines.append(" | ".join(map(str, df.columns)))

        # Rows
        for _, row in df.iterrows():
            values = [
                str(v).strip()
                for v in row
                if pd.notna(v) and str(v).strip()
            ]
            lines.append(" | ".join(values))

        return "\n".join(lines)

    nlp = spacy.load("en_core_web_lg")
    nlpT = spacy.load("en_core_web_trf")

    layout = spaCyLayout(nlp, display_table=display_table)
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

    # Store in `app.config`
    app.config['NLP'] = nlp
    app.config['NLPT'] = nlpT
    app.config['LAYOUT'] = layout
    app.config['SKILL_EXTRACTOR'] = skill_extractor


@app.before_request
def set_global_resources():
    """Attach preloaded models to `g` for request-level access."""
    g.nlp = app.config['NLP']
    g.nlpT = app.config['NLPT']
    g.layout = app.config['LAYOUT']
    g.skill_extractor = app.config['SKILL_EXTRACTOR']
    if current_user.is_authenticated:
        g.llm_api_key = current_user.llm_api_key
    else:
        g.llm_api_key = None


@app.route('/parse', methods=['POST'])
@login_required
def parse():
    resumes = []
    for file in request.files.getlist('resumes'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        resumes.append(filepath)

    resume_str_list, resume_block_list = get_resume_text(resumes, g.layout)

    parsed_data = parse_resume(
        resume_str_list, resume_block_list, g.skill_extractor, g.nlpT)

    # 🔐 Store per-user file using user ID or email
    user_identifier = str(current_user.id)

    user_parsed_json = f"parsed_data_{user_identifier}.json"

    append_to_json(user_parsed_json, parsed_data)

    return jsonify({
        'status': 'success',
        'message': 'Resumes parsed successfully! You can upload more resumes if needed.'
    })


@app.route('/rank', methods=['POST'])
@login_required
def rank():
    user_identifier = str(current_user.id)

    parsed_file = f"parsed_data_{user_identifier}.json"
    results_file = f"results_{user_identifier}.json"

    # print(parsed_file)

    job_role = request.form['job_role']
    jd_text = request.form['jd']
    exp_weight = int(request.form['exp_weight'])
    edu_weight = int(request.form['edu_weight'])
    skill_weight = int(request.form['skill_weight'])

    # print(job_role,jd_text,exp_weight)

    # Store in session for form repopulation
    session['jd_text'] = jd_text
    session['exp_weight'] = exp_weight
    session['edu_weight'] = edu_weight
    session['skill_weight'] = skill_weight

    # Check if parsing is done
    if not os.path.exists(parsed_file):
        session['alert_message'] = "Error: Parsing must be completed before ranking resumes."
        return redirect(url_for('dashboard'))

    try:
        with open(parsed_file, 'r') as f:
            parsed_resumes = json.load(f)

        if not parsed_resumes:
            session['alert_message'] = "Error: No parsed resumes found. Please upload resumes first."
            return redirect(url_for('dashboard'))
    except json.JSONDecodeError:
        session['alert_message'] = "Error: Parsed data is corrupted. Please re-upload resumes."
        return redirect(url_for('dashboard'))

    # Parse skills from job description
    jd_skills = [skill.strip()
                 for skill in jd_text.split(',') if skill.strip()]

    # Get minimum education score
    min_edu_score = None
    if request.form.get('min_edu_score') == 'custom_set' and request.form.get('custom_min_edu_score'):
        min_edu_score = float(request.form.get('custom_min_edu_score'))
    elif request.form.get('min_edu_score') and request.form.get('min_edu_score') != 'custom':
        min_edu_score = float(request.form.get('min_edu_score'))

    # Get required degrees
    required_degrees = None
    if request.form.get('required_degrees'):    
        try:
            required_degrees = json.loads(request.form.get('required_degrees'))
            if not isinstance(required_degrees, list):
                required_degrees = None
        except json.JSONDecodeError:
            required_degrees = None

    # Prepare resume data for ranking
    ranking_data = [
        {
            "name": resume["name"],
            "experience": resume.get("experience", []),
            "education": resume.get("education", []),
            "skills": resume.get("skills", [])
        }
        for resume in parsed_resumes
    ]

    # Rank resumes
    ranked_resumes = rank_resumes(
        ranking_data, jd_skills, exp_weight, edu_weight, skill_weight,
        min_edu_score, required_degrees
    )

    # Load existing results or create new structure
    all_results = {}
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        try:
            with open(results_file, 'r', encoding="utf-8") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = {"screenings": {}}
    else:
        all_results = {"screenings": {}}

    # Generate unique screening ID (timestamp + role)
    screening_id = f"{int(time.time())}_{job_role.replace(' ', '_')}"

    # Add new screening results
    if "screenings" not in all_results:
        all_results["screenings"] = {}

    all_results["screenings"][screening_id] = {
        "job_role": job_role,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "criteria": {
            "skills": jd_skills,
            "exp_weight": exp_weight,
            "edu_weight": edu_weight,
            "skill_weight": skill_weight,
            "min_edu_score": min_edu_score,
            "required_degrees": required_degrees
        },
        "ranked_resumes": json.loads(json.dumps(ranked_resumes, default=lambda o: int(o) if isinstance(o, np.integer) else o))
    }

    # Save all results back to file
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    # Store the current screening ID in session to show it by default
    session['current_screening_id'] = screening_id

    flash("Ranking completed successfully!", "success")
    return redirect(url_for('dashboard_rank'))


@app.route('/all_rank')
@login_required
def all_rank():
    # Get screening_id from URL parameters
    screening_id = request.args.get('screening_id')

    # Use user ID to create a unique file name
    user_identifier = str(current_user.id)
    json_file = f"results_{user_identifier}.json"
    json_file_1 = f"parsed_data_{user_identifier}.json"

    screenings = {}
    ranked_resumes = []
    current_job_role = ""
    current_screening_id = ""

    # Load screening results
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        try:
            with open(json_file, 'r', encoding="utf-8") as f:
                results = json.load(f)
                screenings = results.get('screenings', {})

                # Determine which screening to display
                if screenings:
                    if screening_id and screening_id in screenings:
                        # Use the specified screening_id from URL
                        current_screening = screenings[screening_id]
                        current_screening_id = screening_id
                    else:
                        # If no valid screening_id provided, use the latest one
                        latest_id = sorted(screenings.keys(), key=lambda x: int(
                            x.split('_')[0]), reverse=True)[0]
                        current_screening = screenings[latest_id]
                        current_screening_id = latest_id

                    # Get data for the selected screening
                    ranked_resumes = current_screening.get(
                        'ranked_resumes', [])
                    current_job_role = current_screening.get('job_role', "")
        except json.JSONDecodeError:
            pass

    # Load parsed resumes data
    parsed_resumes = []
    if os.path.exists(json_file_1) and os.path.getsize(json_file_1) > 0:
        try:
            with open(json_file_1, 'r', encoding="utf-8") as f:
                parsed_resumes = json.load(f)
        except json.JSONDecodeError:
            pass

    return render_template('all_rank.html',
                           parsed_resumes=parsed_resumes,
                           ranked_resumes=ranked_resumes,
                           screenings=screenings,
                           current_job_role=current_job_role,
                           current_screening_id=current_screening_id,
                           alert_message=None,
                           form_data={})


@app.route('/dashboard_rank')
@login_required
def dashboard_rank():
    alert_message = session.pop('alert_message', None)

    # Use user ID to create a unique file name
    user_identifier = str(current_user.id)
    json_file = f"results_{user_identifier}.json"
    json_file_1 = f"parsed_data_{user_identifier}.json"

    # Get the current screening ID from session or request
    current_screening_id = request.args.get(
        'screening_id') or session.get('current_screening_id')

    # Form data for repopulating form
    form_data = {
        'jd_text': session.get('jd_text', ''),
        'exp_weight': session.get('exp_weight', 5),
        'edu_weight': session.get('edu_weight', 3),
        'skill_weight': session.get('skill_weight', 4),
    }

    screenings = {}
    ranked_resumes = []
    current_job_role = ""

    # Load screening results
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        try:
            with open(json_file, 'r', encoding="utf-8") as f:
                results = json.load(f)
                screenings = results.get('screenings', {})

                # If we have screenings and a current ID, load that screening
                if screenings and current_screening_id and current_screening_id in screenings:
                    current_screening = screenings[current_screening_id]
                    ranked_resumes = current_screening.get(
                        'ranked_resumes', [])
                    current_job_role = current_screening.get('job_role', "")
                # Otherwise, load the most recent screening if any exist
                elif screenings:
                    # Sort by timestamp (first part of the key)
                    latest_id = sorted(screenings.keys(), key=lambda x: int(
                        x.split('_')[0]), reverse=True)[0]
                    current_screening = screenings[latest_id]
                    ranked_resumes = current_screening.get(
                        'ranked_resumes', [])
                    current_job_role = current_screening.get('job_role', "")
                    session['current_screening_id'] = latest_id
        except json.JSONDecodeError:
            screenings = {}
            ranked_resumes = []

    # Load parsed resumes data
    parsed_resumes = []
    if os.path.exists(json_file_1) and os.path.getsize(json_file_1) > 0:
        try:
            with open(json_file_1, 'r', encoding="utf-8") as f:
                parsed_resumes = json.load(f)
        except json.JSONDecodeError:
            parsed_resumes = []

    return render_template('dashboard_rank.html',
                           parsed_resumes=parsed_resumes,
                           ranked_resumes=ranked_resumes,
                           screenings=screenings,
                           current_job_role=current_job_role,
                           current_screening_id=session.get(
                               'current_screening_id', ''),
                           alert_message=alert_message,
                           form_data=form_data)


@app.route('/clear_screening/<screening_id>')
@login_required
def clear_screening(screening_id):
    user_identifier = str(current_user.id)
    json_file = f"results_{user_identifier}.json"

    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        try:
            with open(json_file, 'r', encoding="utf-8") as f:
                results = json.load(f)

            if 'screenings' in results and screening_id in results['screenings']:
                # Remove the specific screening
                del results['screenings'][screening_id]

                # Write back the updated data
                with open(json_file, 'w') as f:
                    json.dump(results, f, indent=4)

                # If we deleted the current screening, clear the session var
                if session.get('current_screening_id') == screening_id:
                    session.pop('current_screening_id', None)

                flash("Screening results deleted successfully.", "success")
            else:
                flash("Screening not found.", "error")
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
    else:
        flash("No screening results found.", "error")

    return redirect(url_for('dashboard_rank'))


@app.route('/clear_all_screenings')
@login_required
def clear_all_screenings():
    user_identifier = str(current_user.id)
    json_file = f"results_{user_identifier}.json"

    if os.path.exists(json_file):
        try:
            # Create fresh structure with empty screenings
            with open(json_file, 'w') as f:
                json.dump({"screenings": {}}, f, indent=4)

            # Clear current screening from session
            if 'current_screening_id' in session:
                session.pop('current_screening_id', None)

            flash("All screening results cleared successfully.", "success")
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
    else:
        flash("No screening results file found.", "error")

    return redirect(url_for('dashboard_rank'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/resume/<int:index>')
@login_required
def resume_detail(index):
    """Display detailed resume information along with ranking details."""
    user_id = str(current_user.id)
    parsed_file = (f'parsed_data_{user_id}.json')
    ranking_file = (f'results_{user_id}.json')

    if not os.path.exists(parsed_file):
        return redirect(url_for('dashboard'))  # Redirect if no resumes exist

    with open(parsed_file, 'r') as f:
        parsed_resumes = json.load(f)  # Load resumes

    if index < 0 or index >= len(parsed_resumes):
        return "Resume not found", 404  # Handle invalid index

    # Get the resume
    resume = parsed_resumes[index]

    # Load ranking details
    ranking = None
    matched_skills = []
    skill_breakdown = {}

    if os.path.exists(ranking_file):
        with open(ranking_file, 'r') as f:
            ranked_resumes = json.load(f)

        # Find the ranking info for this resume
        ranking = next((r for r in ranked_resumes.get(
            "ranked_resumes", []) if r["index"] == index), None)

        if ranking:
            # Get matched skills if available
            if 'matched_skills' in ranking:
                matched_skills = ranking['matched_skills']

            # Get skill breakdown from ranking if available
            if 'skill_breakdown' in ranking and ranking['skill_breakdown']:
                skill_breakdown = ranking['skill_breakdown']

    # Either use the existing skill_breakdown or the one from ranking
    if not 'skill_breakdown' in resume or not resume['skill_breakdown']:
        resume['skill_breakdown'] = skill_breakdown

    # print(f"Skill breakdown for chart: {resume['skill_breakdown']}")

    return render_template('resume_detail.html',
                           resume=resume,
                           ranking=ranking,
                           matched_skills=matched_skills)


@app.route('/clear_results')
@login_required
def clear_results():
    """Clears session and stored results for the current user."""
    session.pop('ranked_resumes', None)

    user_id = str(current_user.id)
    result_file = (f'results_{user_id}.json')
    parsed_file = (f'parsed_data_{user_id}.json')

    if os.path.exists(result_file):
        os.remove(result_file)
    # if os.path.exists(parsed_file):
    #     os.remove(parsed_file)

    flash("Results cleared successfully.", "info")
    return redirect(url_for('dashboard_rank'))


@app.route('/delete_candidate/<int:index>', methods=['POST'])
@login_required
def delete_candidate(index):
    user_id = str(current_user.id)
    parsed_file = f'parsed_data_{user_id}.json'

    if os.path.exists(parsed_file):
        with open(parsed_file, 'r') as f:
            data = json.load(f)

        if 0 <= index < len(data):
            del data[index]

            with open(parsed_file, 'w') as f:
                json.dump(data, f, indent=4)

    result_file = (f'results_{user_id}.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    return redirect(url_for('dashboard_candidate'))


if __name__ == '__main__':
    with app.app_context():
        load_resources()

    app.run(debug=True, port=5001)

# Call the function to load resources once at startup
