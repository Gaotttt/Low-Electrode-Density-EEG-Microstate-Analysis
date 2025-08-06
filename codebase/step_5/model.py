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
    """Get all .set files in directory and subdirectories."""
    set_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".set"):
                set_files.append(os.path.join(root, file))
    return set_files


def interpolate_segmentation(segmentation, peak_indices, n_times):
    """Interpolate segmentation results back to original time length."""
    interp_func = interp1d(peak_indices, segmentation, kind='linear',
                           bounds_error=False, fill_value=-1)
    return interp_func(np.arange(n_times)).astype(int)


def smooth_segmentation(segmentation, window_size):
    """Apply smoothing to the segmentation."""
    if window_size <= 0:
        return segmentation

    smoothed = np.copy(segmentation)
    for i in range(window_size, len(segmentation) - window_size):
        if segmentation[i] == -1:
            continue
        window = segmentation[i - window_size:i + window_size + 1]
        valid_window = window[window != -1]
        if valid_window.size > 0:
            smoothed[i] = np.bincount(valid_window).argmax()
    return smoothed


def average_consecutive_count(segmentation):
    """Calculate average duration of microstate segments."""
    results = {}
    length_sum_results = {}
    unique_values = np.unique(segmentation)
    unique_values = unique_values[unique_values != -1]  # exclude -1

    for value in unique_values:
        indices = np.where(segmentation == value)[0]
        if not indices.size:
            results[value] = 0
            length_sum_results[value] = 0
            continue

        diff = np.diff(indices)
        segments = np.split(indices, np.where(diff != 1)[0] + 1)
        lengths = [len(s) for s in segments]
        length_sum = sum(lengths)
        avg_length = np.mean(lengths) if lengths else 0
        length_sum_results[value] = length_sum
        results[value] = avg_length

    return results, length_sum_results


def calculate_microstate_metrics(segmentation, sfreq, n_times, n_clusters):
    """Calculate various microstate metrics."""
    metrics = {}

    # Durations
    durations, duration_sums = average_consecutive_count(segmentation)
    metrics['Duration'] = np.array([durations.get(i, 0) / sfreq for i in range(n_clusters)])
    duration_sum_list = [duration_sums.get(i, 0) for i in range(n_clusters)]

    # Occurrences
    counts = {i: 0 for i in range(n_clusters)}
    prev = None
    for val in segmentation:
        if val != -1 and val != prev:
            counts[val] += 1
        prev = val
    metrics['Occurrence'] = np.array([c / (n_times / sfreq) for c in counts.values()])

    # Coverage
    total = sum(duration_sum_list)
    metrics['Coverage'] = np.array([s / total if total > 0 else 0 for s in duration_sum_list])

    # Transition probabilities
    trans_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(len(segmentation) - 1):
        curr, next_ = segmentation[i], segmentation[i + 1]
        if curr != -1 and next_ != -1 and curr != next_:
            trans_matrix[curr, next_] += 1

    metrics['Transition Probabilities'] = (
        trans_matrix / trans_matrix.sum() if trans_matrix.sum() > 0
        else trans_matrix
    )

    return metrics


def compute_metrics(gfp_peaks_data, raw_data, cluster_centers, subject_id,
                    sfreq, peak_indices, window_size=10, lambda_penalty=0):
    """Compute all metrics with optional smoothing and penalty."""
    n_times = raw_data.shape[1]
    n_clusters = cluster_centers.shape[0]

    # Normalize cluster centers
    cluster_centers = cluster_centers / np.linalg.norm(
        cluster_centers, axis=1, keepdims=True)

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

    # Interpolate and smooth
    full_segmentation = interpolate_segmentation(segmentation, peak_indices, n_times)
    full_segmentation = smooth_segmentation(full_segmentation, window_size)

    # Calculate metrics
    metrics = calculate_microstate_metrics(
        full_segmentation, sfreq, n_times, n_clusters)
    metrics['GEV'] = gev

    # Apply transition penalty if specified
    if lambda_penalty > 0:
        n_transitions = np.sum(np.diff(full_segmentation) != 0)
        metrics['GEV'] -= lambda_penalty * n_transitions / len(full_segmentation)

    # Create summary DataFrame
    duration_avg = metrics['Duration'].mean()
    occurrence_avg = 1 / duration_avg if duration_avg > 0 else 0

    summary = {
        'subject_id': [subject_id],
        'GEV': [metrics['GEV']],
        'Duration_avg': [duration_avg],
        'Occurrence_avg': [occurrence_avg]
    }

    # Add individual state metrics
    for i in range(n_clusters):
        summary.update({
            f'Duration_{i + 1}': [metrics['Duration'][i]],
            f'Occurrence_{i + 1}': [metrics['Occurrence'][i]],
            f'Coverage_{i + 1}': [metrics['Coverage'][i]]
        })

    # Add transition probabilities
    for i in range(n_clusters):
        for j in range(n_clusters):
            summary[f'Transition_{i + 1}*to*{j + 1}'] = [
                metrics['Transition Probabilities'][i, j]
            ]

    return full_segmentation, pd.DataFrame(summary)


def run_group_clustering(subject_ids, channels_to_keep):
    """Run group-level clustering on all subjects."""
    all_centers = []
    individual_clusters = []
    gev_values = []

    for subject_id in subject_ids:
        # Load and preprocess data
        raw = read_raw_eeglab(subject_id, preload=True)
        raw.pick("eeg")
        raw.pick_channels(channels_to_keep)

        # Extract GFP peaks
        gfp_peaks = extract_gfp_peaks(raw)

        # Subject-level clustering
        modk = ModKMeans(n_clusters=4, random_state=42)
        modk.fit(gfp_peaks, n_jobs=2)

        gev_values.append(modk.GEV_)
        individual_clusters.append(modk.cluster_centers_)

    # Group-level clustering
    group_centers = np.vstack(individual_clusters).T
    group_centers = ChData(group_centers, modk.info)

    modk_all = ModKMeans(n_clusters=4, random_state=42)
    modk_all.fit(group_centers, n_jobs=2)

    return modk_all, gev_values, individual_clusters, group_centers


def run_app():
    app = FastAPI()

    @app.post("/")
    async def get_answer(request: Request):
        global window_size, lambda_penalty
        request_dict = await request.json()

        raw_data = request_dict.get("raw_data")
        output_dir = request_dict.get("output_dir")
        channels_to_keep = request_dict.get("channels_to_keep")

        # check raw_data
        if not raw_data:
            print("Warning: No input paths provided. Using the default path.")
            raw_data = "/home/medicine/test_data"

        # check output_dir
        if not output_dir:
            print("Warning: No output paths provided. Using the default path.")
            output_dir = "/home/medicine/output"

        if not channels_to_keep:
            print("Warning: No channels_to_keep provided. Using the default value.")
            channels_to_keep = ['C3', 'Cz', 'P2', 'FC2', 'Pz', 'CP1', 'PO3', 'PO4']

        # Create output_dir directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get all .set files in the directory
        subject_ids_all = get_set_files(raw_data)

        print("Running group-level clustering...")
        modk_all, gev_values, indiv_clusters, group_centers = run_group_clustering(subject_ids_all, channels_to_keep)

        # Save clustering results
        with open(os.path.join(output_dir, 'ModK_few_channels.pkl'), 'wb') as f:
            pickle.dump(modk_all, f)
        fig1 = modk_all.plot()
        fig1.savefig(os.path.join(output_dir, 'clusterFig.png'))

        # Save GEV values to Excel
        pd.DataFrame(gev_values).to_excel(
            os.path.join(output_dir, 'gev_values.xlsx'), index=False)


        # Process each file
        all_results_df = pd.DataFrame()
        our_ms_data_all = []

        for subject_id in subject_ids_all:
            # print(f"Processing {subject_id}")

            # Load and preprocess data
            raw = read_raw_eeglab(subject_id, preload=True)
            raw.pick("eeg")
            raw.pick_channels(channels_to_keep)
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
                gfp_peaks_data, raw_data, modk_all.cluster_centers_,
                os.path.basename(subject_id), sfreq, peak_indices,
                window_size, lambda_penalty
            )

            our_ms_data_all.append(segmentation)
            all_results_df = pd.concat([all_results_df, metrics_summary_df])

        # Save results
        csv_filename = os.path.join(output_dir, 'metrics.csv')
        pkl_filename = os.path.join(output_dir, 'segmentation_labels.pkl')

        all_results_df.to_csv(csv_filename, index=False)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(our_ms_data_all, file)


        return {"message": "Microstate metrics have been successfully extracted.",
                "image_path": os.path.join(output_dir, 'clusterFig.png'),
                "metrics_path": os.path.join(output_dir, 'metrics.csv'),
                "segmentation_labels_path": os.path.join(output_dir, 'segmentation_labels.pkl')}

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)


if __name__ == '__main__':
    # Load model
    window_size = 10
    lambda_penalty = 0.0

    # Start FastAPI server
    run_app()
