# Step_6 差异可视化

## 功能
本工具旨在根据输入的脑电微状态指标，得到最后的ICC结果。

## 接口调用

```shell
curl -X 'POST' \
'http://localhost:8080/' \
-H 'Content-Type: application/json' \
-d '{"csv_file_1": "/home/medicine/csv_data/60.csv","csv_file_2": "/home/medicine/csv_data/8.csv","output_dir":"/home/medicine/output"}'
```

### 关键代码
```python
def run_app():
    app = FastAPI()
    @app.post("/")
    async def get_answer(request: Request):
       


       return {"message": "ICC have been successfully visualized.",
                "ICC_image_path": os.path.join(output_dir, 'reliability.svg')}

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
```

### 输入
request字段:

csv_file_1  60通道微状态指标文件
csv_file_2  8通道微状态指标文件
output_dir 指定输出保存的路径


```json
{"csv_file_1": "/home/medicine/csv_data/60.csv",
  "csv_file_2": "/home/medicine/csv_data/8.csv",
  "output_dir":"/home/medicine/output"
}

```

### 输出
response 
"message": "ICC have been successfully visualized.",
"ICC_image_path": os.path.join(output_dir, 'reliability.svg'),

可视化结果 reliability.svg 被保存在指定路径下了

```json
[
  "message": "ICC have been successfully visualized.", 
  "ICC_image_path": os.path.join(output_dir, 'reliability.svg')
]
```
