from flask import Flask, request, jsonify
from analysis_code import analyse_games_mle, verify_games_data
import pandas as pd
import io
from flask_cors import CORS


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


def format_output(
    players,
    win_ratios,
    theta,
    pairwise_differences,
    pairwise_std,
    significant,
):
    pairwise_table = [
        [
            (
                None
                if i == j
                else {
                    "diff": pairwise_differences[i, j],
                    "std": pairwise_std[i, j],
                    "significant": bool(significant[i, j]),
                }
            )
            for j in range(len(players))
        ]
        for i in range(len(players))
    ]

    statements = [
        f"{players[i]} is significantly more likely to win than {players[j]}"
        for i in range(len(players))
        for j in range(len(players))
        if i != j and significant[i, j] and pairwise_differences[i, j] > 0
    ]

    return jsonify(
        {
            "players": players.tolist(),
            "skill_scores": [float(s) for s in theta],
            "win_ratios": [float(r) for r in win_ratios],
            "pairwise_table": pairwise_table,
            "significant_statements": statements,
        }
    )


@app.post("/analyze")
def analyze():
    file = request.files.get("file")
    alpha = float(request.form.get("alpha", 0.05))

    if not file or not file.filename:
        return jsonify({"error": "No data uploaded"}), 400

    try:
        games = read_file(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read data: {str(e)}"}), 400

    try:
        verify_games_data(games)
    except Exception as e:
        return jsonify({"error": f"Dataset is invalid: {str(e)}"}), 400

    try:
        (
            players,
            win_ratios,
            theta,
            pairwise_differences,
            pairwise_std,
            significant,
        ) = analyse_games_mle(games, alpha)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    return format_output(
        players,
        win_ratios,
        theta,
        pairwise_differences,
        pairwise_std,
        significant,
    )


if __name__ == "__main__":
    app.run(debug=True)
