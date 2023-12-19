@echo off
set CONDA_ENV_NAME="E:\Projects\py3.8\venv"
rem set PYTHON_SCRIPT=file_name.py

echo Activating Conda environment: %CONDA_ENV_NAME%
call conda activate %CONDA_ENV_NAME%

echo Prase html/xml to plain text...
call python py_3.8.py

rem echo Running Python script: %PYTHON_SCRIPT%
rem python %PYTHON_SCRIPT%

echo Deactivating Conda environment: %CONDA_ENV_NAME%
call conda deactivate
