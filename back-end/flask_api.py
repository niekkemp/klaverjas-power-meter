from flask import Flask, request, jsonify
from analysis_code import analyze_games, verify_games_data
import pandas as pd
import io
from flask_cors import CORS
import time


app = Flask(__name__)
CORS(app)


def read_file(file):
    if file.filename.lower().endswith(".csv"):
        data_str = file.read().decode("utf-8")
        games = pd.read_csv(io.StringIO(data_str)).to_numpy()
    else:
        file.seek(0)
        games = pd.read_excel(file).to_numpy()
    return games


@app.post("/analyze")
def analyze():
    # time.sleep(1)
    # load params with defaults
    file = request.files.get("file")
    alpha = float(request.form.get("alpha", 0.05))
    method = str(request.form.get("method", "mle"))
    tau = float(request.form.get("tau", 1.0))

    # ensure file was provided
    if not file or not file.filename:
        return jsonify({"error": "No dataset provided"}), 400

    # read the file
    try:
        games = read_file(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read data: {str(e)}"}), 400

    # verify data is compatitable with analysis
    try:
        verify_games_data(games)
    except Exception as e:
        return jsonify({"error": f"Dataset is invalid: {str(e)}"}), 400

    # do the analysis
    try:
        result = analyze_games(games, alpha, method, tau)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
