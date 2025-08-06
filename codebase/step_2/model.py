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


def compute_gev(gfp_data, cluster_centers):
    """
    Compute the Global Explained Variance (GEV) for given GFP data and cluster centers.
    Parameters
    ----------
    gfp_data : ndarray
        The GFP data, shape (1, n_samples).
    cluster_centers : ndarray
        The cluster centers (microstates), shape (n_clusters, n_channels).
    Returns
    -------
    gev : float
        The Global Explained Variance.
    """

    # Step 1: Ensure GFP data and cluster centers are properly shaped
    gfp_data = gfp_data.squeeze()
    n_samples = gfp_data.shape[0]
    n_clusters, n_channels = cluster_centers.shape

    # Step 2: Standardize the GFP data
    gfp_sum_sq = np.sum(gfp_data**2)

    # Step 3: Ensure cluster centers are normalized
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    # Step 4: Compute projection values (activation) and matching microstates (segmentation)
    activation = np.dot(cluster_centers, gfp_data)
    segmentation = np.argmax(np.abs(activation), axis=0)

    # Step 5: Compute data and centroid correlation (map_corr)
    map_corr = np.array([np.corrcoef(gfp_data[:, t], cluster_centers[segmentation[t]])[0, 1] for t in range(gfp_data.shape[1])])
    map_corr = np.abs(map_corr)

    # Step 6: Compute Global Explained Variance (GEV)
    gev = np.sum((gfp_data * map_corr) ** 2) / gfp_sum_sq

    return gev, gfp_sum_sq, activation, segmentation, map_corr


def run_app():
    app = FastAPI()

    @app.post("/")
    async def get_answer(request: Request):
        global channels_to_drop
        request_dict = await request.json()

        raw_data = request_dict.get("raw_data")
        output_dir = request_dict.get("output_dir")

        # check raw_data
        if not raw_data:
            print("Warning: No input paths provided. Using the default path.")
            raw_data = "/home/medicine/test_data"

        # check output_dir
        if not output_dir:
            print("Warning: No output paths provided. Using the default path.")
            output_dir = "/home/medicine/output"

        # Create output_dir directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get all .set files in the directory
        subject_ids_all = get_set_files(raw_data)

        all_centers = list()
        individual_cluster_centers_all = list()
        all = []
        for subject_id in subject_ids_all:
            # load Data
            raw = read_raw_eeglab(subject_id, preload=True)
            raw.pick("eeg")
            # channels_to_drop = ['TP9', 'TP10', 'HEOG', 'VEOG']
            raw.drop_channels(channels_to_drop)
            # extract GFP peaks
            gfp_peaks = extract_gfp_peaks(raw)
            # subject level clustering
            ModK = ModKMeans(n_clusters=4, random_state=42)
            ModK.fit(gfp_peaks, n_jobs=2)
            all.append(ModK.GEV_)
            individual_cluster_centers_all.append(ModK.cluster_centers_)

        group_cluster_centers_all = np.vstack(individual_cluster_centers_all).T
        group_cluster_centers_all = ChData(group_cluster_centers_all, ModK.info)

        # group level clustering
        ModK_all = ModKMeans(n_clusters=4, random_state=42)
        ModK_all.fit(group_cluster_centers_all, n_jobs=2)
        ModK_all.plot()
        # print(ModK_all.GEV_)
        # print(ModK_all.tol)
        all_centers.append(ModK_all.cluster_centers_)

        with open(os.path.join(output_dir, 'ModK_all.pkl'), 'wb') as file:
            pickle.dump(ModK_all, file)
        # print(f"对象已保存到 {os.path.join(output_dir, 'ModK_all.pkl')}")

        with open(os.path.join(output_dir, 'ModK_all.pkl'), 'rb') as f:
            ModK_allall = pickle.load(f)
        fig1 = ModK_allall.plot()
        fig1.savefig(os.path.join(output_dir, 'clusterFig.png'))

        # gev
        our_results_all = list()
        our_ms_data_all = list()
        for subject_id in subject_ids_all:
            # Load Data
            raw_HC = read_raw_eeglab(subject_id, preload=True)
            raw_HC.pick("eeg")
            # channels_to_drop = ['TP9', 'TP10', 'HEOG', 'VEOG']
            raw_HC.drop_channels(channels_to_drop)
            gfp_peaks = extract_gfp_peaks(raw_HC)
            gev, gfp_sum_sq, activation, segmentation, map_corr = compute_gev(gfp_peaks.get_data(),
                                                                              ModK_all.cluster_centers_)
            our_ms_data_all.append(segmentation)
            print(gev)
            our_results_all.append(gev)
        df_all = pd.DataFrame(our_results_all)
        excel_file_path = os.path.join(output_dir, 'our_output_all.xlsx')
        with pd.ExcelWriter(excel_file_path) as writer:
            df_all.to_excel(writer, sheet_name='Sheet1', index=False)


        return {"message": "Microstate clustering has been completed.",
                "image_path": os.path.join(output_dir, 'clusterFig.png')}

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)

if __name__ == '__main__':
    # Load model
    channels_to_drop = ['TP9', 'TP10', 'HEOG', 'VEOG']

    # Start FastAPI server
    run_app()
