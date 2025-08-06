# Step_4 脑电微状态聚类工具说明

## 功能
本工具旨在根据输入的脑电数据和分段标签，训练深度模型，得出通道重要性结论。

## 接口调用

```shell
curl -X 'POST' \
'http://localhost:8080/' \
-H 'Content-Type: application/json' \
-d '{"raw_data":"/home/medicine/test_data",
"label_file":"/home/medicine/test_labels/segmentation_labels.pkl",
"output_dir":"/home/medicine/output",
"sampling_rate":500,
"overlap":0,
"batch_size":2,
"total_epoch":2,
"input_channel_num":60,
}'

curl -X 'POST' \
'http://localhost:8080/' \
-H 'Content-Type: application/json' \
-d '{"raw_data": "/home/medicine/test_data",
"label_file":"/home/medicine/test_labels/segmentation_labels.pkl",
"output_dir":"/home/medicine/output",
"batch_size":2,
"total_epoch":2
}'

```

### 关键代码
```python
def run_app():
    app = FastAPI()
    @app.post("/")
    async def get_answer(request: Request):
        


        return {"message": "Successfully.",
                "log_path": os.path.join(output_dir, 'runs'),
                "checkpoint_path": os.path.join(output_dir, 'checkpoints'),
                "results_path": os.path.join(output_dir, 'grad_cam_results')}
    
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
```

### 输入
request字段:

raw_data  指定输入路径（包含多个.set格式文件的脑电数据路径）
label_file 指定标签文件
output_dir 指定输出保存的路径
sampling_rate 脑电采样率 
overlap 脑电切片重叠百分比取值范围[0-1）
batch_size 模型训练的批次大小
total_epoch 模型训练的轮次
input_channel_num 脑电数据的通道数量
channel_names 脑电通道顺序列表

```json
{
 "raw_data": "/home/medicine/test_data",
"label_file":"/home/medicine/test_labels/segmentation_labels.pkl",
"output_dir":"/home/medicine/output",
"sampling_rate":500,
"overlap":0,
"batch_size":2,
"total_epoch":2,
"input_channel_num":60,
"channel_names":['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2',
                'FC5', 'FC6', 'CP5', 'CP6', 'Fpz', 'FCz', 'CPz', 'POz', 'F1', 'F2', 'C1', 'C2',
                'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6',
                'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8']
}
```

### 输出
response 
"message": "Successfully."
"log_path": os.path.join(output_dir, 'runs')
"checkpoint_path": os.path.join(output_dir, 'checkpoints')
"results_path": os.path.join(output_dir, 'grad_cam_results')

训练过程记录 log_path; 训练保存的参数 checkpoint_path 其他结果都被保存在results_path路径下了

```json
[
  "message": "Successfully.",
                "log_path": os.path.join(output_dir, 'runs'),
                "checkpoint_path": os.path.join(output_dir, 'checkpoints'),
                "results_path": os.path.join(output_dir, 'grad_cam_results')
]
```
