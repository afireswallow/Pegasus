@echo off

for %%i in (8 16 20 32 36 48) do (
    python codegen.py mlp%%i.p4g new-mlp%%i.p4
)
pause
