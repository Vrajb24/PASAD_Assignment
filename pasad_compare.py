import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def pasad_compare(file_path, N=2000, L=24, start_train_idx=0, end_train_idx=2000, test_start_idx=2000):
    """
    Perform PASAD analysis using SSA and compare two projection methods (UT and UUT).

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

    # Load the dataset
    data = pd.read_csv(file_path)

    # Split the data into training and testing sets
    train_data = data.iloc[start_train_idx:end_train_idx, 1].values  # Using measurements for training
    test_data = data.iloc[test_start_idx:, 1].values                 # Using remaining data for testing

    # Display basic information
    print(f"Training data length: {len(train_data)}")
    print(f"Testing data length: {len(test_data)}")

    # Create the trajectory matrix for training data
    K = N - L + 1  # Number of lagged vectors
    trajectory_matrix = np.zeros((L, K))

    # Construct the lagged vectors for the trajectory matrix
    for i in range(K):
        trajectory_matrix[:, i] = train_data[i:i + L]

    # Perform Singular Value Decomposition on the trajectory matrix
    U, Sigma, VT = np.linalg.svd(trajectory_matrix, full_matrices=False)

    # Select the top 'r' components based on singular values (choosing r=1 for simplicity)
    r = 1
    U_r = U[:, :r]

    # Project the lagged training vectors onto the signal subspace
    projected_train_vectors = U_r.T @ trajectory_matrix

    # Compute the cluster_mean of the projected vectors
    cluster_mean = np.mean(projected_train_vectors, axis=1)

    # Function to calculate departure score using the UT method
    def compute_departure_score_ut(test_vector, U_r, cluster_mean):
        projected_test_vector = U_r.T @ test_vector
        departure_score = np.linalg.norm(projected_test_vector - cluster_mean)**2
        return departure_score

    # Function to calculate departure score using the UUT method
    def compute_departure_score_uut(test_vector, U_r, cluster_mean):
        projected_test_vector = U_r @ (U_r.T @ test_vector)
        departure_score = np.linalg.norm(projected_test_vector - U_r @ cluster_mean)**2
        return departure_score

    # Create test vectors and compute departure scores for both methods
    test_departure_scores_ut = []
    test_departure_scores_uut = []

    # Time tracking for both methods
    ut_times = []
    uut_times = []

    for i in range(len(test_data) - L + 1):
        test_vector = test_data[i:i + L]
        
        # Compute UT method departure score
        start_time = time.time()
        score_ut = compute_departure_score_ut(test_vector, U_r, cluster_mean)
        ut_times.append(time.time() - start_time)
        test_departure_scores_ut.append(score_ut)

        # Compute UUT method departure score
        start_time = time.time()
        score_uut = compute_departure_score_uut(test_vector, U_r, cluster_mean)
        uut_times.append(time.time() - start_time)
        test_departure_scores_uut.append(score_uut)

    # Convert scores to numpy arrays for easier handling
    test_departure_scores_ut = np.array(test_departure_scores_ut)
    test_departure_scores_uut = np.array(test_departure_scores_uut)

    # Calculate the threshold for attack detection based on UT method
    threshold_ut = np.mean(test_departure_scores_ut) + 2 * np.std(test_departure_scores_ut)
    threshold_uut = np.mean(test_departure_scores_uut) + 2 * np.std(test_departure_scores_uut)

    # Detect attack points for both methods
    attack_indices_ut = np.where(test_departure_scores_ut > threshold_ut)[0]
    attack_indices_uut = np.where(test_departure_scores_uut > threshold_uut)[0]

    # Measure average runtimes for UT and UUT methods
    avg_ut_time = np.mean(ut_times)
    avg_uut_time = np.mean(uut_times)

    print(f"Average runtime for UT method: {avg_ut_time * 1000:.5f} ms")
    print(f"Average runtime for UUT method: {avg_uut_time * 1000:.5f} ms")

    # Plot departure scores and highlight detected attacks for both methods
    plt.figure(figsize=(14, 10))

    # Plot for UT method
    plt.subplot(2, 1, 1)
    plt.plot(test_departure_scores_ut, label='Departure Scores (UT)', color='orange')
    plt.axhline(y=threshold_ut, color='green', linestyle='--', label=f'Threshold = {threshold_ut:.2f}')
    plt.scatter(attack_indices_ut, test_departure_scores_ut[attack_indices_ut], color='red', label='Detected Attacks (UT)')
    plt.title('Departure Scores with Attack Detection (UT Method)')
    plt.xlabel('Time Index')
    plt.ylabel('Departure Score')
    plt.legend()

    # Plot for UUT method
    plt.subplot(2, 1, 2)
    plt.plot(test_departure_scores_uut, label='Departure Scores (UUT)', color='orange')
    plt.axhline(y=threshold_uut, color='green', linestyle='--', label=f'Threshold = {threshold_uut:.2f}')
    plt.scatter(attack_indices_uut, test_departure_scores_uut[attack_indices_uut], color='red', label='Detected Attacks (UUT)')
    plt.title('Departure Scores with Attack Detection (UUT Method)')
    plt.xlabel('Time Index')
    plt.ylabel('Departure Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print out the indices where attacks were detected
    if len(attack_indices_ut) > 0:
        print(f"Attack detected using UT method at time indices (relative to test data): {attack_indices_ut}")
    else:
        print("No attacks detected using UT method.")

    if len(attack_indices_uut) > 0:
        print(f"Attack detected using UUT method at time indices (relative to test data): {attack_indices_uut}")
    else:
        print("No attacks detected using UUT method.")
