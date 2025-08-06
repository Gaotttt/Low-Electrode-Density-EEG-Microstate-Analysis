# Step_2 脑电微状态聚类工具说明

## 功能
本工具旨在对输入的脑电数据进行微状态聚类，输出聚类结果、聚类模板图和GEV表格。

## 接口调用

```shell
curl -X 'POST' \
'http://localhost:8080/' \
-H 'Content-Type: application/json' \
-d '{"raw_data":"/home/medicine/test_data",
"output_dir":"/home/medicine/output"
}'
```

### 关键代码
```python
def run_app():
    app = FastAPI()
    @app.post("/")
        async def get_answer(request: Request):
        global channels_to_drop
        request_dict = await request.json()

        raw_data = request_dict.get("raw_data")
        output_dir = request_dict.get("output_dir")

        # 检查 raw_data 是否为空
        if not raw_data:
            return ["error: No input paths provided. Using the default path."]
        raw_data = "/home/medicine/test_data"

        # 检查 output_dir 是否为空
        if not output_dir:
            return ["error: No output paths provided. Using the default path."]
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
        # print(f"数据已成功保存到 {excel_file_path}")

        return {"message": "Microstate clustering has been completed."}

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
```

### 输入
request字段:

raw_data  指定输入路径（包含多个.set格式文件的脑电数据路径）
output_dir 指定输出保存的路径


```json
{
  "raw_data": "/home/medicine/test_data",
  "output_dir":"/home/medicine/output"
}
```

### 输出
response 就是一个提示："message": "Microstate clustering has been completed."
聚类结果 ModK_all.pkl; 聚类模板图 clusterFig.png GEV表格 our_output_all.xlsx 都被保存在指定路径下了

```json
[
  "message": "Microstate clustering has been completed.",
  "image_path": os.path.join(output_dir, 'clusterFig.png')
]
```
