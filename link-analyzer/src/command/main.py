from flask.cli import load_dotenv
import os
from flask_app_factory import create_app


def main():
    load_dotenv()
    app = create_app()
    HOST = os.getenv('ANALYSIS_SERVER_HOST')
    PORT = os.getenv('ANALYSIS_SERVER_PORT')
    app.run(HOST, PORT, debug=False)

if __name__ == "__main__":
    main()

