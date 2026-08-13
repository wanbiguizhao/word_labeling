"""启动汉字切割标注系统服务器"""
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."]
    )
