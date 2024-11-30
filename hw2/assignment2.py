#################################
# Your name: Dor Liberman
#################################
import random

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        samples = np.zeros((m, 2))
        for i in range(m):
            # Sample x uniformly from [0, 1]
            x = random.uniform(0, 1)
            # Determine P[y=1|x]
            if (0.0 <= x <= 0.2) or (0.4 <= x <= 0.6) or (0.8 <= x <= 1.0):
                Py1 = 0.8
            else:
                Py1 = 0.1
            # Sample y based on P[y=1|x]
            y = 1 if random.uniform(0, 1) < Py1 else 0
            samples[i, 0] = x
            samples[i, 1] = y
        return samples

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        results = []
        m_values = range(m_first, m_last + 1, step)

        for m in m_values:
            empirical_errors = []
            true_errors = []
            for _ in range(T):
                # Step (i): Draw a sample of size m
                sample = self.sample_from_D(m)
                xs = sample[:, 0]
                ys = sample[:, 1]
                # Sort the sample based on x
                sorted_indices = np.argsort(xs)
                xs_sorted = list(xs[sorted_indices])
                ys_sorted = list(ys[sorted_indices])

                # Step (ii): Run the ERM algorithm
                # The find_best_interval function is assumed to return (intervals, error_count)
                intervals_found, _ = intervals.find_best_interval(xs_sorted, ys_sorted, k)

                # Step (iii): Calculate the empirical error
                # Predictions based on the intervals
                predictions = np.zeros(m, dtype=int)
                for (l, u) in intervals_found:
                    # Vectorized assignment for efficiency
                    mask = (xs_sorted >= l) & (xs_sorted <= u)
                    predictions[mask] = 1
                empirical_error = np.mean(predictions != ys_sorted)
                empirical_errors.append(empirical_error)

                # Step (iv): Calculate the true error
                true_error = self.compute_true_error(intervals_found)
                true_errors.append(true_error)

            # Calculate average errors over T trials
            avg_empirical = np.mean(empirical_errors)
            avg_true = np.mean(true_errors)
            results.append([avg_empirical, avg_true])

            print(f"m = {m}: Avg Empirical Error = {avg_empirical:.4f}, Avg True Error = {avg_true:.4f}")

        # Convert results to numpy array for easier handling
        results_array = np.array(results)

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(m_values, results_array[:, 0], label='Average Empirical Error', marker='o')
        plt.plot(m_values, results_array[:, 1], label='Average True Error', marker='s')
        plt.xlabel('Sample Size (m)')
        plt.ylabel('Error')
        plt.title(f'ERM: Empirical and True Errors vs Sample Size (k={k})')
        plt.legend()
        plt.grid(True)
        plt.show()

        return results_array

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # Draw a sample of size m
        sample = self.sample_from_D(m)
        xs = sample[:, 0]
        ys = sample[:, 1]
        sorted_indices = np.argsort(xs)
        xs_sorted = list(xs[sorted_indices])
        ys_sorted = list(ys[sorted_indices])

        empirical_errors = []
        true_errors = []
        ks = list(range(k_first, k_last + 1, step))

        for k in ks:
            # Find the best hypothesis for current k
            intervals_found, _ = intervals.find_best_interval(xs_sorted, ys_sorted, k)

            # Calculate empirical error
            predictions = np.zeros(m, dtype=int)
            for (l, u) in intervals_found:
                mask = (xs_sorted >= l) & (xs_sorted <= u)
                predictions[mask] = 1
            empirical_error = np.mean(predictions != ys_sorted)
            empirical_errors.append(empirical_error)

            # Calculate true error
            true_error = self.compute_true_error(intervals_found)
            true_errors.append(true_error)

        # Plot empirical and true errors
        plt.figure(figsize=(10, 6))
        plt.plot(ks, empirical_errors, label="Empirical Error", marker="o")
        plt.plot(ks, true_errors, label="True Error", marker="s")
        plt.xlabel("Number of Intervals (k)")
        plt.ylabel("Error")
        plt.title("Empirical and True Errors vs Number of Intervals (k)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Find k* with the smallest empirical error
        k_star = ks[np.argmin(empirical_errors)]
        print(f"k* (smallest empirical error) = {k_star}")

        return k_star

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        sample = self.sample_from_D(m)
        xs = sample[:, 0]
        ys = sample[:, 1]
        sorted_indices = np.argsort(xs)
        xs_sorted = list(xs[sorted_indices])
        ys_sorted = list(ys[sorted_indices])

        empirical_errors = []
        penalties = []
        srm_errors = []
        true_errors = []
        ks = list(range(k_first, k_last + 1, step))
        n = m  # Sample size

        for k in ks:
            # Find the best hypothesis for current k
            intervals_found, _ = intervals.find_best_interval(xs_sorted, ys_sorted, k)

            # Calculate empirical error
            predictions = np.zeros(m, dtype=int)
            for (l, u) in intervals_found:
                mask = (xs_sorted >= l) & (xs_sorted <= u)
                predictions[mask] = 1
            empirical_error = np.mean(predictions != ys_sorted)
            empirical_errors.append(empirical_error)

            # Calculate penalty
            VCdim_Hk = 2 * k  # VC dimension of H_k
            delta_k = 0.1 / (k ** 2)
            penalty = np.sqrt((2 * VCdim_Hk + np.log(2 / delta_k)) / n)
            penalties.append(penalty)

            # Calculate SRM error (empirical error + penalty)
            srm_error = empirical_error + penalty
            srm_errors.append(srm_error)

            # Calculate true error
            true_error = self.compute_true_error(intervals_found)
            true_errors.append(true_error)

        # Plot empirical error, penalty, SRM error, and true error
        plt.figure(figsize=(10, 6))
        plt.plot(ks, empirical_errors, label="Empirical Error", marker="o")
        plt.plot(ks, penalties, label="Penalty", marker="s")
        plt.plot(ks, srm_errors, label="Empirical + Penalty (SRM)", marker="^")
        plt.plot(ks, true_errors, label="True Error", marker="x")
        plt.xlabel("Number of Intervals (k)")
        plt.ylabel("Error")
        plt.title("SRM: Empirical Error, Penalty, True Error, and SRM Error vs Number of Intervals (k)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Find k_srm with the smallest SRM error
        k_srm = ks[np.argmin(srm_errors)]
        print(f"k_srm (smallest SRM error) = {k_srm}")

        return k_srm

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # Step 1: Draw a dataset and split into training and validation sets
        data = self.sample_from_D(m)
        np.random.shuffle(data)  # Shuffle to ensure random split
        train_size = int(0.8 * m)
        train_data = data[:train_size]
        validation_data = data[train_size:]

        # Split into xs and ys for training and validation
        xs_train = train_data[:, 0]
        ys_train = train_data[:, 1]
        xs_val = validation_data[:, 0]
        ys_val = validation_data[:, 1]

        sorted_indices_train = np.argsort(xs_train)
        xs_train_sorted = list(xs_train[sorted_indices_train])
        ys_train_sorted = list(ys_train[sorted_indices_train])

        validation_errors = []
        true_errors = []
        ks = list(range(1, 11))  # k ranges from 1 to 10

        for k in ks:
            # Step 2: Train on the training set (find ERM hypothesis)
            intervals_found, _ = intervals.find_best_interval(xs_train_sorted, ys_train_sorted, k)

            # Step 3: Evaluate on the validation set
            predictions_val = np.zeros(len(xs_val), dtype=int)
            for (l, u) in intervals_found:
                mask = (xs_val >= l) & (xs_val <= u)
                predictions_val[mask] = 1
            validation_error = np.mean(predictions_val != ys_val)
            validation_errors.append(validation_error)

            # Step 4: Calculate true error for this hypothesis
            true_error = self.compute_true_error(intervals_found)
            true_errors.append(true_error)

        # Step 5: Find the best k based on validation error
        k_holdout = ks[np.argmin(validation_errors)]
        print(f"k_holdout (lowest validation error) = {k_holdout}")

        # Plot validation errors and true errors
        plt.figure(figsize=(10, 6))
        plt.plot(ks, validation_errors, label="Validation Error", marker="o")
        plt.plot(ks, true_errors, label="True Error", marker="x")
        plt.xlabel("Number of Intervals (k)")
        plt.ylabel("Error")
        plt.title("Holdout-Validation: Validation and True Errors vs Number of Intervals (k)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return k_holdout

    def compute_true_error(self, intervals_list):
        """
        Calculates the true error e_P(h_I) for a given hypothesis h_I.

        Parameters:
        - intervals_list: List of tuples [(l1, u1), (l2, u2), ..., (lk, uk)]

        Returns:
        - total_error: The true error e_P(h_I)
        """
        # Define the regions where P[y=1|x] changes
        change_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Add interval endpoints to the change points
        for (l, u) in intervals_list:
            change_points.append(l)
            change_points.append(u)

        # Remove duplicates and sort
        sorted_points = sorted(list(set(change_points)))

        total_error = 0.0

        # Iterate through each sub-interval
        for i in range(len(sorted_points) - 1):
            a = sorted_points[i]
            b = sorted_points[i + 1]
            # Midpoint to determine P[y=1|x] in this sub-interval
            midpoint = (a + b) / 2.0

            # Determine P[y=1|x] based on midpoint
            if (0.0 <= midpoint <= 0.2) or (0.4 <= midpoint <= 0.6) or (0.8 <= midpoint <= 1.0):
                Py1 = 0.8
            else:
                Py1 = 0.1

            # Determine h_I(x) in this sub-interval
            hI = 0  # Default prediction
            for (l, u) in intervals_list:
                if l <= midpoint <= u:
                    hI = 1
                    break

            # Calculate error in this sub-interval
            if hI == 1:
                error = (1 - Py1) * (b - a)  # Predicting 1, error if y=0
            else:
                error = Py1 * (b - a)  # Predicting 0, error if y=1

            total_error += error

        return total_error


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
