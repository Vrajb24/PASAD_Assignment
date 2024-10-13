import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def pasad_ssa_mahalanobis(file_path, N=2000, L=24, start_train_idx=0, end_train_idx=2000, test_start_idx=2000):
    """
    Perform PASAD analysis using SSA and display scree plot, raw test data plot, and departure scores.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the dataset.
    N : int, optional
        Number of initial samples for training (default is 2000).
    L : int, optional
        Lag parameter for trajectory matrix (default is 24).
    start_train_idx : int, optional
        Start index for the training data (default is 0).
    end_train_idx : int, optional
        End index for the training data (default is 2000).
    test_start_idx : int, optional
        Start index for the test data (default is 2000).

    Returns:
    --------
    None
    """

    start_time = time.time()  # Start time tracking

    # Load the dataset
    data = pd.read_csv(file_path)

    # Split the data into training and testing sets
    train_data = data.iloc[start_train_idx:end_train_idx, 1].values  # Using measurements for training
    test_data = data.iloc[test_start_idx:, 1].values                 # Using remaining data for testing

    # Create the trajectory matrix for training data
    K = N - L + 1  # Number of lagged vectors
    trajectory_matrix = np.zeros((L, K))

    # Construct the lagged vectors for the trajectory matrix
    for i in range(K):
        trajectory_matrix[:, i] = train_data[i:i + L]

    # Perform Singular Value Decomposition on the trajectory matrix
    U, Sigma, VT = np.linalg.svd(trajectory_matrix, full_matrices=False)

    # Plot the significance of each component (Scree plot)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(Sigma) + 1), Sigma, 'o-', label='Singular Values')
    plt.title('Scree Plot - Significance of Each Component')
    plt.xlabel('Component Index (r)')
    plt.ylabel('Singular Value (Significance)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Select the top 'r' components based on singular values (choosing r=2 for this case)
    r = 2
    U_r = U[:, :r]

    # Project the lagged training vectors onto the signal subspace
    projected_train_vectors = U_r.T @ trajectory_matrix

    # Compute the cluster_mean of the projected vectors
    cluster_mean = np.mean(projected_train_vectors, axis=1)

    # Compute the covariance matrix of the projected training vectors
    covariance_matrix = np.cov(projected_train_vectors)

    # Precompute the inverse of the covariance matrix for Mahalanobis distance
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)

    # Function to calculate departure score for a new test vector using Mahalanobis distance
    def compute_departure_score(test_vector, U_r, cluster_mean, inv_cov_matrix):
        # Project the test vector onto the signal subspace
        projected_test_vector = U_r.T @ test_vector
        # Compute the Mahalanobis distance from the cluster_mean
        diff = projected_test_vector - cluster_mean
        departure_score = diff.T @ inv_cov_matrix @ diff
        return departure_score

    # Create test vectors and compute departure scores
    test_departure_scores = []
    for i in range(len(test_data) - L + 1):
        test_vector = test_data[i:i + L]
        score = compute_departure_score(test_vector, U_r, cluster_mean, inv_covariance_matrix)
        test_departure_scores.append(score)

    # Convert scores to a numpy array for easier plotting
    test_departure_scores = np.array(test_departure_scores)

    # Calculate the threshold for attack detection
    threshold = np.mean(test_departure_scores) + 2 * np.std(test_departure_scores)

    # Detect attack points: indices where departure scores exceed the threshold
    attack_indices = np.where(test_departure_scores > threshold)[0]

    # Plot the raw test data and highlight the training, test, and attack regions
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_data, label='Training Data', color='blue')
    plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_data, label='Test Data', color='orange')

    # If attacks are detected, highlight attack regions
    if len(attack_indices) > 0:
        for attack_start in attack_indices:
            plt.axvspan(len(train_data) + attack_start, len(train_data) + attack_start + L, color='red', alpha=0.3)
    plt.title('Raw Test Data')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.legend()

    # Plot the departure scores and mark the detected attack regions
    plt.subplot(2, 1, 2)
    plt.plot(test_departure_scores, label='Departure Scores', color='orange')
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f}')

    # Highlight the attack points in the departure score plot
    plt.scatter(attack_indices, test_departure_scores[attack_indices], color='red', label='Detected Attacks')
    plt.title('Departure Scores with Attack Detection')
    plt.xlabel('Time Index')
    plt.ylabel('Departure Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print out the indices where attacks were detected
    if len(attack_indices) > 0:
        print(f"Attack detected at time indices (relative to test data): {attack_indices}")
    else:
        print("No attacks detected.")

    # Final end time after all operations
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

# Example usage
# pasad_ssa_mahalanobis(file_path="path_to_your_dataset.csv", N=800, L=24)
