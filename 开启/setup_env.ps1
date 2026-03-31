# Windows / WSL2 环境下安装 Python 环境并安装依赖

Write-Host "创建 Conda 环境 ulm_mar..."
conda create -n ulm_mar python=3.10 -y
Write-Host "激活环境..."
conda activate ulm_mar
Write-Host "升级 pip 并安装依赖..."
python -m pip install --upgrade pip
pip install -r .\requirements.txt
Write-Host "安装完成。请使用 'conda activate ulm_mar' 激活环境，然后运行 uvicorn app:app --reload."