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


def sigmoid(log_odds: np.ndarray) -> np.ndarray:
    """
    Logistic function mapping log-odds to probabilities.
    """
    return 1 / (1 + np.exp(-log_odds))


def neg_log_likelihood(theta: np.ndarray, X: np.ndarray) -> float:
    """
    Negative log-likelihood for the logistic skill model:
    p(win) = sigmoid(sum_winners - sum_losers).
    """
    log_odds = X @ theta
    probs = sigmoid(log_odds)
    return -np.sum(np.log(probs + 1e-12))


def neg_log_likelihood_grad(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Gradient of the negative log-likelihood.
    """
    log_odds = X @ theta
    probs = sigmoid(log_odds)
    return -X.T @ (1 - probs)


def neg_log_likelihood_hessian(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Hessian (observed Fisher information) of the negative log-likelihood.
    """
    log_odds = X @ theta
    probs = sigmoid(log_odds)
    W = np.diag(probs * (1 - probs))
    return X.T @ W @ X


# estimation


def estimate_player_skills(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    theta_init = np.zeros(n_players - 1)

    # maximum likelihood estimation
    result = minimize(
        lambda theta: neg_log_likelihood(theta, X_reduced),
        theta_init,
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
    theta_cov_centered = C @ theta_full_cov @ C.T

    return theta_centered, theta_cov_centered


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
    n_players = theta_cov.shape[0]
    std_matrix = np.zeros_like(theta_cov)
    for i in range(n_players):
        for j in range(n_players):
            var_diff = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
            std_matrix[i, j] = np.sqrt(var_diff)
    return std_matrix


def check_significance(
    pairwise_differences: np.ndarray, pairwise_std: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Tests pairwise skill differences for significance at given alpha
    using normal approximation.
    """
    z_crit = norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        significant = np.abs(pairwise_differences) > z_crit * pairwise_std
    np.fill_diagonal(significant, False)
    return significant


# analysis


def analyse_games_mle(
    games: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates player skill scores θ_j from 2v2 match results using a logistic model:

        p(winning team wins | θ) = sigmoid(sum_winners - sum_losers)

    A maximum-likelihood estimate is computed with one player's skill fixed
    as a reference, then shifted to set the mean skill to zero. Pairwise skill
    differences and their standard errors are reported, with significance testing
    at a user-specified alpha level.

    All outputs are ordered consistently with the final `players` ordering.

    Parameters
    ----------
    games : np.ndarray
        An (n_games x 4) array, where each row contains the four players
        involved in a game:
        - first two columns: winners
        - last two columns: losers

    alpha : float, optional
        Significance level for testing pairwise skill differences
        (default is 0.05).

    Returns
    -------
    players : np.ndarray
        Array of player identifiers, ordered by estimated skill scores (ties
        broken by win ratio if needed).

    win_ratios : np.ndarray
        Observed win ratios for each player.

    theta : np.ndarray
        Estimated skill scores for each player, normalized to have mean zero.

    pairwise_differences : np.ndarray
        Matrix of pairwise skill differences: entry (i, j) is theta_i - theta_j.

    pairwise_std : np.ndarray
        Standard errors of the pairwise differences.

    significant : np.ndarray
        Boolean matrix of the same shape as `pairwise_differences`, where
        True indicates a significant difference at the given alpha level.
    """
    # extract necessary info from games data
    players, win_ratios, X = process_games(games)

    # set theta of first player to zero for reference
    theta, theta_cov = estimate_player_skills(X)

    # sort by skill, break ties by win ratio
    sort_idx = np.lexsort((-win_ratios, -theta))
    players = players[sort_idx]
    theta = theta[sort_idx]
    theta_cov = theta_cov[sort_idx, :][:, sort_idx]

    # compute pairwise skill differences
    pairwise_differences = compute_pairwise_differences(theta)
    pairwise_std = compute_pairwise_std(theta_cov)
    significant = check_significance(pairwise_differences, pairwise_std, alpha)

    return (
        players,
        win_ratios,
        theta,
        pairwise_differences,
        pairwise_std,
        significant,
    )


# test script


if __name__ == "__main__":
    file_path = "./back-end/example_data.csv"
    alpha = 0.1
    games = load_games_data(file_path)
    print(games)
    verify_games_data(games)

    players, theta, win_ratios, pairwise_differences, pairwise_std, significant = (
        analyse_games_mle(games, alpha)
    )

    print("Leaderboard (skill with win ratio):")
    for player, skill, ratio in zip(players, theta, win_ratios):
        print(f" - {player}: {skill:.2f} (win ratio: {ratio:.2%})")

    n_players = len(players)
    formatted = np.empty((n_players, n_players), dtype=object)
    for i in range(n_players):
        for j in range(n_players):
            if i == j:
                formatted[i, j] = "-"
            else:
                diff = pairwise_differences[i, j]
                std = pairwise_std[i, j]
                star = "*" if significant[i, j] else ""
                formatted[i, j] = f"{diff:.2f} ({std:.2f}){star}"

    df = pd.DataFrame(formatted, index=players, columns=players)
    print(
        f"\nPairwise skill log-odds differences (std) [* significant at alpha={alpha}]:"
    )

    print(df)

    print(f"\nSignificant pairwise skill advantages (alpha={alpha}):")
    found = False
    for i in range(n_players):
        for j in range(n_players):
            if i != j and significant[i, j] and pairwise_differences[i, j] > 0:
                print(
                    f" - {players[i]} is significantly more likely to win than {players[j]}"
                )
                found = True
    if not found:
        print(" (none found)")
