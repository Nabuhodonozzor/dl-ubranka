@ECHO OFF
ECHO activating venv
call ..\venv\Scripts\activate

ECHO running script
python sources\predict.py

ECHO deactivating venv
deactivate

