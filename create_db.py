from app import app, db  # Import app and db from your main Flask file

# Ensure this runs within the app context
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully.")
