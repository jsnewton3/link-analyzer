import os
from flask import Flask
from controllers.monitoring_task_session_handler import Monitor_Task_Session_Handler
session_handler = Monitor_Task_Session_Handler()

def create_app():
    # Init the core app
    # load_dotenv()
    app = Flask(__name__)
    # Set the config parameters from the environment variables
    app.config['FLASK_DEBUG'] = os.getenv('ANALYSIS_FLASK_DEBUG')
    app.config['TESTING'] = os.getenv('ANALYSIS_FLASK_TESTING')
    with app.app_context():
        # Import routes
        from routes.routes import monitor_blueprint
        # Register Blueprint
        app.register_blueprint(monitor_blueprint)
    return app