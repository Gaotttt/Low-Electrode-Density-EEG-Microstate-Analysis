# Step_3 脑电微状态聚类工具说明

## 功能
本工具旨在根据输入的脑电数据以及输入的微状态聚类模板，输出微状态指标结果和分段标签。

## 接口调用

```shell
curl -X 'POST' \
'http://localhost:8080/' \
-H 'Content-Type: application/json' \
-d '{"raw_data":"/home/medicine/test_data","template_file":"/home/medicine/test_template/ModK_all_reorder_60.pkl","output_dir":"/home/medicine/output"}'
```

### 关键代码
```python
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
        template_file = "/home/medicine/test_template"

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
        our_ms_data_all = []

        for subject_id in subject_ids_all:
            # print(f"Processing {subject_id}")

            # Load and preprocess data
            raw = read_raw_eeglab(subject_id, preload=True)
            raw.pick("eeg")
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

            our_ms_data_all.append(segmentation)
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
```

### 输入
request字段:

raw_data  指定输入路径（包含多个.set格式文件的脑电数据路径）
template_file 指定微状态模板文件
output_dir 指定输出保存的路径


```json
{
  "raw_data": "/home/medicine/test_data",
  "output_dir":"/home/medicine/output"
}
```

### 输出
response 
"message": "Microstate metrics have been successfully extracted."
"metrics_path": os.path.join(output_dir, 'metrics.csv'),
"segmentation_labels_path": os.path.join(output_dir, 'segmentation_labels.pkl')

微状态指标 metrics.csv; 分段标签 segmentation_labels.pkl 都被保存在指定路径下了

```json
[
  "message": "Microstate metrics have been successfully extracted.",
  "metrics_path": os.path.join(output_dir, 'metrics.csv'),
  "segmentation_labels_path": os.path.join(output_dir, 'segmentation_labels.pkl')
]
```
