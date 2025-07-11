import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# data I/O and verification


def load_games_data(file_path: str) -> np.ndarray:
    """
    Load csv games data via file path.
    """
    return pd.read_csv(file_path).to_numpy()


def verify_games_data(games: np.ndarray):
    """
    Verify the games data to ensure it's compatitable with the analysis.
    """
    if games.shape[0] == 0:
        raise ValueError("Dataset is empty")
    if games.shape[1] != 4:
        raise ValueError(
            f"Expected 4 columns (2 winners, 2 losers), got {games.shape[1]}"
        )
    if not np.all([all(isinstance(x, str) for x in game) for game in games]):
        raise ValueError("Dataset contains empty values")
    if not all(len(set(game)) == 4 for game in games):
        raise ValueError("Games contain duplicate players")


# preprocessing


def process_games(games: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a games matrix to extract player statistics and build a design matrix.

    Each row of `games` should list:
    - winner1, winner2, loser1, loser2

    Parameters
    ----------
    games : np.ndarray
        An (n_games x 4) array where each row contains the players participating in a game,
        with the first two columns as winners and the last two as losers.

    Returns
    -------
    players : np.ndarray
        Sorted array of all unique player identifiers.

    win_ratios : np.ndarray
        Array of win ratios for each player, ordered consistently with `players`.

    X : np.ndarray
        The design matrix (n_games x n_players), with columns ordered consistently
        with `players`, where for each row:
        - winners have +1 in their column
        - losers have -1 in their column
        - all other entries are zero
    """

    # extract unique players and their play counts
    players, play_counts = np.unique(games, return_counts=True)
    n_players = len(players)
    n_games = games.shape[0]

    # build a consistent player-to-index map
    player_to_index = {player: idx for idx, player in enumerate(players)}
    player_to_index_vec = np.vectorize(player_to_index.get)

    # compute win counts
    winner_players, winner_counts = np.unique(games[:, :2], return_counts=True)
    win_counts = np.zeros(n_players, dtype=int)
    win_counts[player_to_index_vec(winner_players)] = winner_counts

    # calculate win ratios per player
    win_ratios = win_counts / play_counts

    # build the design matrix:
    # +1 for winners, -1 for losers, 0 otherwise
    game_indices = player_to_index_vec(games)  # (n_games x 4) indices
    X = np.zeros((n_games, n_players), dtype=int)

    # mark winners
    for i in range(2):
        X[np.arange(n_games), game_indices[:, i]] = 1
    # mark losers
    for i in range(2, 4):
        X[np.arange(n_games), game_indices[:, i]] = -1

    return players, win_ratios, X


# core stats utils


def sigmoid(x):
    """
    Logistic function mapping log-odds to probabilities.
    """
    return 1 / (1 + np.exp(-x))


def neg_log_likelihood(
    theta: np.ndarray, X: np.ndarray, tau_squared: float = 0.0
) -> float:
    """
    Negative log-likelihood for logistic skill model with optional L2 regularization.
    """
    log_odds = X @ theta
    probs = sigmoid(log_odds)
    nll = -np.sum(np.log(probs + 1e-12))  # avoid log(0)

    if tau_squared > 0:
        penalty = 0.5 / tau_squared * np.dot(theta, theta)
    else:
        penalty = 0.0

    return nll + penalty


def neg_log_likelihood_grad(
    theta: np.ndarray, X: np.ndarray, tau_squared: float = 0.0
) -> np.ndarray:
    """
    Gradient of the negative log-likelihood with optional L2 regularization.
    """
    log_odds = X @ theta
    probs = sigmoid(log_odds)
    grad = X.T @ (probs - 1)

    if tau_squared > 0:
        grad += (1 / tau_squared) * theta

    return grad


def neg_log_likelihood_hessian(
    theta: np.ndarray, X: np.ndarray, tau_squared: float = 0.0
) -> np.ndarray:
    """
    Hessian of the negative log-likelihood with optional L2 regularization.
    """
    log_odds = X @ theta
    probs = sigmoid(log_odds)
    W = np.diag(probs * (1 - probs))
    hessian = X.T @ W @ X

    if tau_squared > 0:
        hessian += (1 / tau_squared) * np.eye(len(theta))

    return hessian


# estimation


def estimate_player_skills_mle(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate player skill scores from the design matrix using a logistic model.

    One player's skill is fixed to zero for identifiability, then all skills are
    shifted to have mean zero. Returns the estimated skills along with their
    centered covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        (n_games x n_players) design matrix, where
        - winners have +1
        - losers have -1
        - others are 0

    Returns
    -------
    theta_centered : np.ndarray
        Estimated skill scores for all players, normalized to have mean zero.

    theta_cov_centered : np.ndarray
        Covariance matrix of the mean-zero skill estimates.
    """
    n_players = X.shape[1]
    # remove the first player column to fix theta_0 = 0
    X_reduced = X[:, 1:]
    theta_reduced_init = np.zeros(n_players - 1)

    # maximum likelihood estimation
    result = minimize(
        lambda theta: neg_log_likelihood(theta, X_reduced),
        theta_reduced_init,
        jac=lambda theta: neg_log_likelihood_grad(theta, X_reduced),
    )
    theta_reduced = result.x

    # compute covariance (inverse Fisher)
    fisher_info = neg_log_likelihood_hessian(theta_reduced, X_reduced)
    theta_reduced_cov = np.linalg.inv(fisher_info)

    # reconstruct full theta (theta_0 = 0) and pad covariance
    theta_full = np.concatenate(([0], theta_reduced))
    theta_full_cov = np.pad(
        theta_reduced_cov, ((1, 0), (1, 0)), mode="constant", constant_values=0
    )

    # center to have mean zero (projection)
    C = np.eye(n_players) - np.ones((n_players, n_players)) / n_players
    theta_centered = C @ theta_full
    theta_centered_cov = C @ theta_full_cov @ C.T

    return theta_centered, theta_centered_cov


def estimate_player_skills_ridge(
    X: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate player skill scores from the design matrix using a logistic model
    with Gaussian prior (Ridge regularization).

    Regularization resolves identifiability without fixing any player.
    Returns the estimated skills along with their centered covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        (n_games x n_players) design matrix, where
        - winners have +1
        - losers have -1
        - others are 0

    tau : float
        Standard deviation of the prior on each player's skill (θ_j ~ N(0, τ²)).
        Smaller τ implies stronger regularization toward zero skill.

    Returns
    -------
    theta_centered : np.ndarray
        Estimated skill scores for all players, normalized to have mean zero.

    theta_cov_centered : np.ndarray
        Covariance matrix of the mean-zero skill estimates.
    """
    if not tau > 0:
        raise ValueError("Tau must be a positive number")

    n_players = X.shape[1]
    theta_init = np.zeros(n_players)

    tau_squared = tau**2

    # maximum likelihood estimation
    result = minimize(
        lambda theta: neg_log_likelihood(theta, X, tau_squared),
        theta_init,
        jac=lambda theta: neg_log_likelihood_grad(theta, X, tau_squared),
    )
    theta = result.x

    # compute covariance (inverse Fisher)
    fisher_info = neg_log_likelihood_hessian(theta, X, tau_squared)
    theta_cov = np.linalg.inv(fisher_info)

    # center to have mean zero (projection)
    C = np.eye(n_players) - np.ones((n_players, n_players)) / n_players
    theta_centered = C @ theta
    theta_centered_cov = C @ theta_cov @ C.T

    return theta_centered, theta_centered_cov


# pairwise utils


def compute_pairwise_differences(theta: np.ndarray) -> np.ndarray:
    """
    Returns pairwise skill differences:
    (i, j) = theta[i] - theta[j].
    """
    return theta[:, None] - theta[None, :]


def compute_pairwise_std(theta_cov: np.ndarray) -> np.ndarray:
    """
    Computes standard errors for all pairwise skill differences
    using the skill covariance matrix.
    """
    var_diff = np.diag(theta_cov)[:, None] + np.diag(theta_cov)[None, :] - 2 * theta_cov
    return np.sqrt(var_diff)


def check_elwise_sig(values: np.ndarray, stds: np.ndarray, alpha: float) -> np.ndarray:
    """
    Tests pairwise skill differences for significance at given alpha
    using normal approximation.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    z_crit = norm.ppf(1 - alpha / 2)
    significant = np.abs(values) > z_crit * stds
    # np.fill_diagonal(significant, False)
    return significant


# formatting the response


def format_response(
    players: np.ndarray,
    win_ratios: np.ndarray,
    theta: np.ndarray,
    theta_std: np.ndarray,
    theta_sig: np.ndarray,
    pairwise_differences: np.ndarray,
    pairwise_std: np.ndarray,
    pairwise_sig: np.ndarray,
    alpha: float,
) -> dict:
    """
    Format player skill and significance results into a JSON-serializable dictionary.

    All NumPy arrays are converted to native Python types (lists, floats, bools),
    making the return value safe for use with `jsonify()` in Flask or `json.dumps()`.

    Returns
    -------
    dict[str, Any]
        A JSON-serializable dictionary with:
        - players: list of player names
        - ranking_table: list of dicts per player with skill estimates
        - statements: list of natural language strings about significant differences
        - pairwise_table: matrix of dicts or None with skill differences and significance
    """
    players = players.tolist()

    ranking_table = [
        {
            "player": player,
            "win_rate": float(win_rate),
            "skill": float(skill),
            "std": float(std),
            "sig": bool(sig),
        }
        for player, win_rate, skill, std, sig in zip(
            players, win_ratios, theta, theta_std, theta_sig
        )
    ]

    statements = [
        f"{players[i]} plays significantly better than {players[j]}."
        for i in range(len(players))
        for j in range(len(players))
        if i != j and pairwise_sig[i, j] and pairwise_differences[i, j] > 0
    ]

    pairwise_table = [
        [
            (
                None
                if i == j
                else {
                    "diff": float(pairwise_differences[i, j]),
                    "std": float(pairwise_std[i, j]),
                    "sig": bool(pairwise_sig[i, j]),
                }
            )
            for j in range(len(players))
        ]
        for i in range(len(players))
    ]

    return {
        "players": players,
        "ranking_table": ranking_table,
        "statements": statements,
        "pairwise_table": pairwise_table,
    }


# analysis


def analyze_games(
    games: np.ndarray, alpha: float = 0.05, method: str = "mle", tau: float = 1.0
) -> dict:
    """
    Analyze 2v2 match results and estimate player skill scores using a logistic model.

    Each game is modeled as a logistic comparison between the sum of winning team
    skills and the sum of losing team skills:
        p(winning team wins | θ) = sigmoid(θ₁ + θ₂ - θ₃ - θ₄)

    Depending on the selected method, skills are inferred using:
    - "mle"   : Maximum Likelihood Estimation (requires fixing one player's skill)
    - "ridge" : Maximum A Posteriori Estimation with Gaussian prior (Ridge)

    Skills are normalized to have zero mean. Pairwise skill differences and their
    standard errors are returned, along with significance tests for each pair.

    The returned dictionary is fully JSON-serializable and suitable for use in
    web applications (e.g. via Flask `jsonify()`).

    Parameters
    ----------
    games : np.ndarray
        (n_games x 4) array. Each row contains 4 player identifiers:
        - first two columns: winners
        - last two columns: losers

    alpha : float, optional
        Significance level for testing pairwise skill differences.
        Must be between zero and one (default: 0.05).

    method : str, optional
        Estimation method: "mle" for unregularized maximum likelihood,
        or "ridge" for MAP estimation with Gaussian prior (default: "mle").

    tau : float, optional
        Standard deviation of the prior (θ_j ~ N(0, τ²)) used in Ridge regression.
        Ignored if method is "mle". Must be positive if used (default: 1.0).

    Returns
    -------
    dict
        A JSON-serializable dictionary with the following keys:
        - 'players'         : list of player identifiers
        - 'ranking_table'   : list of dicts with skill stats per player
        - 'statements'      : list of significant skill comparisons (strings)
        - 'pairwise_table'  : matrix of skill difference dicts (or None on diagonal)
    """
    # extract necessary info from games data
    players, win_ratios, X = process_games(games)

    # set theta of first player to zero for reference
    if method == "mle":
        theta, theta_cov = estimate_player_skills_mle(X)
    elif method == "ridge":
        theta, theta_cov = estimate_player_skills_ridge(X, tau)
    else:
        raise ValueError("Method must be 'mle' or 'ridge'")

    # sort by skill, break ties by win ratio
    sort_idx = np.lexsort((-win_ratios, -theta))
    players = players[sort_idx]
    win_ratios = win_ratios[sort_idx]
    theta = theta[sort_idx]
    theta_cov = theta_cov[sort_idx, :][:, sort_idx]

    # compute significances of difference of skill from average
    theta_std = np.sqrt(np.diagonal(theta_cov))
    theta_sig = check_elwise_sig(theta, theta_std, alpha)

    # compute pairwise skill differences
    pairwise_differences = compute_pairwise_differences(theta)
    pairwise_std = compute_pairwise_std(theta_cov)
    pairwise_sig = check_elwise_sig(pairwise_differences, pairwise_std, alpha)

    return format_response(
        players,
        win_ratios,
        theta,
        theta_std,
        theta_sig,
        pairwise_differences,
        pairwise_std,
        pairwise_sig,
        alpha,
    )


# test script


if __name__ == "__main__":
    import pandas as pd

    file_path = "./back-end/example_data.csv"
    alpha = 0.1
    games = load_games_data(file_path)
    print(games)
    verify_games_data(games)

    response = analyze_games(games, alpha, method="ridge", tau=1)

    players = response["players"]
    ranking_table = response["ranking_table"]
    pairwise_table = response["pairwise_table"]
    statements = response["statements"]

    print("\nLeaderboard (skill with win ratio):")
    for row in ranking_table:
        print(
            f" - {row['player']}: {row['skill']:.2f} "
            f"(win ratio: {row['win_rate']:.2%}, std: {row['std']:.2f})"
            + (" *" if row["sig"] else "")
        )

    print(
        f"\nPairwise skill log-odds differences (std) "
        f"[* significant at alpha={alpha}]:"
    )

    n_players = len(players)
    formatted = []
    for i in range(n_players):
        row = []
        for j in range(n_players):
            if i == j:
                row.append("-")
            else:
                entry = pairwise_table[i][j]
                if entry["sig"]:
                    row.append(f"{entry['diff']:.2f} ({entry['std']:.2f})*")
                else:
                    row.append(f"{entry['diff']:.2f} ({entry['std']:.2f})")
        formatted.append(row)

    df = pd.DataFrame(formatted, index=players, columns=players)
    print(df)

    print(f"\nSignificant pairwise skill advantages (alpha={alpha}):")
    if statements:
        for stmt in statements:
            print(" -", stmt)
    else:
        print(" (none found)")
