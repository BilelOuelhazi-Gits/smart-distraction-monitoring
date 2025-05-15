from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pymongo import MongoClient
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['attention_db']
collection = db['distraction_time']

# Hardcoded users
USERS = {
    'admin': {'password': 'admin', 'role': 'admin'},
    'user': {'password': 'user', 'role': 'user'}
}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = USERS.get(username)

        if user and user['password'] == password:
            session['username'] = username
            session['role'] = user['role']
            return redirect(url_for(f"{user['role']}_dashboard"))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return "Access denied"
    return render_template('admin_dashboard.html', username=session.get('username'))


@app.route('/admin_dashboard/data')
def admin_dashboard_data():
    if session.get('role') != 'admin':
        return jsonify({"error": "Access denied"}), 403

    # Fetch user data from MongoDB for 'test_user'
    user_data = collection.find_one({"user": "test_user"}) or {}

    data = {
        "distraction_time": user_data.get('distraction_time', 0),
        "hand_on_mouth_count": user_data.get('hand_on_mouth_count', 0),
        "blink_count": user_data.get('blink_count', 0),
        "posture_alerts": user_data.get('posture_alerts', []),
        "total_phone_usage_time": user_data.get('total_phone_usage_time', 0)
    }

    return jsonify(data)
@app.route('/user_dashboard')
def user_dashboard():
    if session.get('role') != 'user':
        return "Access denied"
    return render_template('user_dashboard.html', username=session.get('username'))

@app.route('/user_dashboard/data')
def user_dashboard_data():
    if session.get('role') != 'user':
        return jsonify({"error": "Access denied"}), 403

    # Fetch user data from MongoDB for 'test_user'
    user_data = collection.find_one({"user": "test_user"}) or {}

    data = {
        "distraction_time": user_data.get('distraction_time', 0),
        "hand_on_mouth_count": user_data.get('hand_on_mouth_count', 0),
        "blink_count": user_data.get('blink_count', 0),
        "posture_alerts": user_data.get('posture_alerts', []),
        "total_phone_usage_time": user_data.get('total_phone_usage_time', 0)
    }

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)