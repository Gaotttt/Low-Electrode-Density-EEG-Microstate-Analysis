import os
import pandas as pd
import numpy as np
from mne.io import read_raw_eeglab
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks
from fastapi import FastAPI, Request
import uvicorn
import pickle
from scipy.interpolate import interp1d


def get_set_files(directory):
    """
    Get a list of all .set files in the given directory.

    Parameters
    ----------
    directory : str
        Path to the directory.

    Returns
    -------
    list of str
        List of .set file paths.
    """
    set_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".set"):
                set_files.append(os.path.join(root, file))
    return set_files


def interpolate_segmentation(segmentation, peak_indices, n_times):
    """Interpolate segmentation results back to original time length."""
    interp_func = interp1d(peak_indices, segmentation, kind='linear', bounds_error=False, fill_value=-1)
    full_segmentation = interp_func(np.arange(n_times)).astype(int)
    return full_segmentation


def compute_metrics(gfp_peaks_data, raw_data, cluster_centers, subject_id, sfreq, peak_indices, window_size=10,
                    lambda_penalty=0):
    """Compute GEV and other metrics with optional smoothing and penalty."""
    n_times = raw_data.shape[1]
    n_clusters = cluster_centers.shape[0]

    # Normalize cluster centers
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    # Compute activation and segmentation at GFP peaks
    activation = np.dot(cluster_centers, gfp_peaks_data)
    segmentation = np.argmax(np.abs(activation), axis=0)

    # Compute correlation between data and microstate centers
    map_corr = np.array([
        np.corrcoef(gfp_peaks_data[:, t], cluster_centers[segmentation[t]])[0, 1]
        for t in range(gfp_peaks_data.shape[1])
    ])
    map_corr = np.abs(map_corr)

    # Compute GEV at GFP peaks
    gfp_sum_sq = np.sum(gfp_peaks_data ** 2, axis=0)
    gev = np.sum((gfp_peaks_data * map_corr) ** 2) / np.sum(gfp_sum_sq)

    # Interpolate back to original time length
    full_segmentation = interpolate_segmentation(segmentation, peak_indices, n_times)

    # Apply smoothing if window_size > 0
    if window_size > 0:
        full_segmentation = smooth_segmentation(full_segmentation, window_size)

    # Calculate microstate metrics
    metrics = calculate_microstate_metrics(full_segmentation, sfreq, n_times, n_clusters)
    metrics['GEV'] = gev

    # Apply penalty for non-smooth transitions if lambda_penalty > 0
    if lambda_penalty > 0:
        metrics['GEV'] -= lambda_penalty * np.sum(np.diff(full_segmentation) != 0) / len(full_segmentation)

    # Create summary DataFrame
    duration_avg = metrics['Duration'].mean()
    occurrence_avg = 1 / duration_avg if duration_avg > 0 else 0

    metrics_summary_df = pd.DataFrame({
        'subject_id': [subject_id],
        'GEV': [metrics['GEV']],
        **{f'Duration_{i + 1}': [metrics['Duration'][i]] for i in range(n_clusters)},
        'Duration_avg': [duration_avg],
        **{f'Occurrence_{i + 1}': [metrics['Occurrence'][i]] for i in range(n_clusters)},
        'Occurrence_avg': [occurrence_avg],
        **{f'Coverage_{i + 1}': [metrics['Coverage'][i]] for i in range(n_clusters)}
    })

    # Add transition probabilities
    for i in range(n_clusters):
        for j in range(n_clusters):
            metrics_summary_df[f'Transition_{i + 1}*to*{j + 1}'] = metrics['Transition Probabilities'][i, j]

    return full_segmentation, metrics_summary_df


def calculate_microstate_metrics(segmentation, sfreq, n_times, n_clusters):
    """Calculate various microstate metrics."""
    metrics = {}

    # Calculate durations
    our_duration, duration_sum = average_consecutive_count(segmentation)
    duration_list = [value / sfreq for value in our_duration.values()]
    duration_sum_list = [value for value in duration_sum.values()]
    metrics['Duration'] = np.array(duration_list)

    # Calculate occurrences
    sequence_counts = {i: 0 for i in range(n_clusters)}
    previous_value = None
    for value in segmentation:
        if value != -1 and value != previous_value:
            sequence_counts[value] += 1
        previous_value = value
    metrics['Occurrence'] = np.array([value / (n_times / sfreq) for value in sequence_counts.values()])

    # Calculate coverage
    metrics['Coverage'] = np.array(duration_sum_list) / sum(duration_sum_list)

    # Calculate transition probabilities
    transition_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(len(segmentation) - 1):
        if segmentation[i] != -1 and segmentation[i + 1] != -1 and segmentation[i] != segmentation[i + 1]:
            transition_matrix[segmentation[i], segmentation[i + 1]] += 1

    metrics[
        'Transition Probabilities'] = transition_matrix / transition_matrix.sum() if transition_matrix.sum() > 0 else transition_matrix

    return metrics


def smooth_segmentation(segmentation, window_size):
    """Apply smoothing to the segmentation."""
    smoothed_segmentation = np.copy(segmentation)
    for i in range(window_size, len(segmentation) - window_size):
        if segmentation[i] == -1:
            continue
        window = segmentation[i - window_size:i + window_size + 1]
        valid_window = window[window != -1]
        if len(valid_window) > 0:
            smoothed_segmentation[i] = np.bincount(valid_window).argmax()
    return smoothed_segmentation


def average_consecutive_count(segmentation):
    """Calculate average duration of microstate segments."""
    results = {}
    length_sum_results = {}
    for value in np.unique(segmentation):
        if value == -1:
            continue
        indices = np.where(segmentation == value)[0]
        if len(indices) == 0:
            results[value] = 0
            length_sum_results[value] = 0
            continue

        diff = np.diff(indices)
        segments = np.split(indices, np.where(diff != 1)[0] + 1)
        lengths = [len(segment) for segment in segments]
        length_sum = sum(lengths)
        average_length = np.mean(lengths) if lengths else 0
        length_sum_results[value] = length_sum
        results[value] = average_length

    return results, length_sum_results


def run_app():
    app = FastAPI()

    @app.post("/")
    async def get_answer(request: Request):
        global channels_to_drop, window_size, lambda_penalty
        request_dict = await request.json()

        raw_data = request_dict.get("raw_data")
        template_file = request_dict.get("template_file")
        output_dir = request_dict.get("output_dir")

        # check raw_data
        if not raw_data:
            print("Warning: No input paths provided. Using the default path.")
            raw_data = "/home/medicine/test_data"

        if not template_file:
            print("Warning: No EEG Microstate Clustering Templates provided. Using the default path.")
            template_file = "/home/medicine/test_template/ModK_all_reorder_60.pkl"

        # check output_dir
        if not output_dir:
            print("Warning: No output paths provided. Using the default path.")
            output_dir = "/home/medicine/output"

        # Create output_dir directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the ModKMeans model
        with open(template_file, 'rb') as f:
            ModK_all = pickle.load(f)

        # Get all .set files in the directory
        subject_ids_all = get_set_files(raw_data)

        # Process each file
        all_results_df = pd.DataFrame()
        our_ms_data_all = {}  # Changed from list to dictionary

        for subject_id in subject_ids_all:
            # print(f"Processing {subject_id}")

            # Load and preprocess data
            raw = read_raw_eeglab(subject_id, preload=True)
            raw.pick("eeg")
            raw.drop_channels(channels_to_drop)
            gfp_peaks = extract_gfp_peaks(raw)

            sfreq = raw.info['sfreq']
            raw_data = raw.get_data()
            gfp_peaks_data = gfp_peaks.get_data()

            # Find peak indices
            peak_indices = []
            for i in range(gfp_peaks_data.shape[1]):
                peak_index = np.where(np.all(raw_data == gfp_peaks_data[:, i][:, np.newaxis], axis=0))[0]
                if len(peak_index) > 0:
                    peak_indices.append(peak_index[0])
            peak_indices = np.array(peak_indices)

            # Compute metrics
            segmentation, metrics_summary_df = compute_metrics(
                gfp_peaks_data, raw_data, ModK_all.cluster_centers_,
                os.path.basename(subject_id), sfreq, peak_indices,
                window_size, lambda_penalty
            )

            our_ms_data_all[os.path.basename(subject_id)] = segmentation
            all_results_df = pd.concat([all_results_df, metrics_summary_df])

        # Save results
        csv_filename = os.path.join(output_dir, 'metrics.csv')
        pkl_filename = os.path.join(output_dir, 'segmentation_labels.pkl')

        all_results_df.to_csv(csv_filename, index=False)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(our_ms_data_all, file)


        return {"message": "Microstate metrics have been successfully extracted.",
                "metrics_path": os.path.join(output_dir, 'metrics.csv'),
                "segmentation_labels_path": os.path.join(output_dir, 'segmentation_labels.pkl')}

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)


if __name__ == '__main__':
    # Load model
    channels_to_drop = ['TP9', 'TP10', 'HEOG', 'VEOG']
    window_size = 10
    lambda_penalty = 0.0

    # Start FastAPI server
    run_app()
