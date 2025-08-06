# Step_5 脑电微状态聚类工具说明

## 功能
本工具旨在根据输入的脑电数据以及指定的通道，输出微状态聚类模板、微状态指标结果、分段标签。

## 接口调用

```shell
curl -X 'POST' \
'http://localhost:8080/' \
-H 'Content-Type: application/json' \
-d '{"raw_data": "/home/medicine/test_data","output_dir":"/home/medne/output","channels_to_keep":["C3","Cz","P2","FC2","Pz","CP1","PO3","PO4"]}'
```

### 关键代码
```python
def run_app():
    app = FastAPI()
    @app.post("/")
    async def get_answer(request: Request):
       


        return {"message": "Microstate metrics have been successfully extracted.",
                "image_path": os.path.join(output_dir, 'clusterFig.png'),
                "metrics_path": os.path.join(output_dir, 'metrics.csv'),
                "segmentation_labels_path": os.path.join(output_dir, 'segmentation_labels.pkl')}

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
```

### 输入
request字段:

raw_data  指定输入路径（包含多个.set格式文件的脑电数据路径）
output_dir 指定输出保存的路径
channels_to_keep 指定微状态通道


```json
{
  "raw_data": "/home/medicine/test_data",
  "output_dir":"/home/medicine/output",
  "channels_to_keep":["C3","Cz","P2","FC2","Pz","CP1","PO3","PO4"]
}
```

### 输出
response 
"message": "Microstate metrics have been successfully extracted.",
"image_path": os.path.join(output_dir, 'clusterFig.png'),
"metrics_path": os.path.join(output_dir, 'metrics.csv'),
"segmentation_labels_path": os.path.join(output_dir, 'segmentation_labels.pkl')

微状态聚类结果 clusterFig.png; 微状态指标 metrics.csv; 分段标签 segmentation_labels.pkl 都被保存在指定路径下了

```json
[
  "message": "Microstate metrics have been successfully extracted.",
  "image_path": os.path.join(output_dir, 'clusterFig.png'),
  "metrics_path": os.path.join(output_dir, 'metrics.csv'),
  "segmentation_labels_path": os.path.join(output_dir, 'segmentation_labels.pkl')
]
```
