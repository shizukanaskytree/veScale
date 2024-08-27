
```bash
/root/vescale_prj/veScale/
│
├── tracing_viewer/
│   ├── logs/
│   │   └── tracing-finetune_4D-20240821_144339.log  # 日志文件
│   │
│   ├── src/
│   │   ├── main.py                  # 启动应用程序（唯一的入口点）
│   │   ├── log_parser.py            # 解析日志文件
│   │   ├── code_snippet_extractor.py # 提取源代码片段
│   │   └── templates/
│   │       └── index.html           # 网页模板
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css           # 样式表
│   │   └── js/
│   │       └── main.js              # JavaScript文件
│
├── vescale/                     # 你的源代码文件夹
│   ├── devicemesh_api/
│   ├── dtensor/
│   ├── debug/
│   └── ...
│
└── README.md                    # 项目文档
```

How to run the website?

1.
```bash
pip install --ignore-installed blinker Flask
```

2.
```bash
cd /root/vescale_prj/veScale/tracing_viewer/src
# Run the Flask application
python main.py
```

3.
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

