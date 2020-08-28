@ECHO OFF
ECHO activatnig venv
call ..\venv\Scripts\activate
ECHO running script
python sources/train_new_model.py
ECHO running finished
deactivate
pause