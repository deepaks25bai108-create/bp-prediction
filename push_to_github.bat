@echo off
title Push BP Prediction to GitHub
echo ========================================================
echo   Pushing Blood Pressure Prediction Project to GitHub
echo   Target: https://github.com/deepaks25bai108-create/bp-prediction.git
echo ========================================================
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Code successfully pushed to GitHub!
) else (
    echo [ERROR] Push failed. Make sure you have created the repository 'bp-prediction' on https://github.com/new
)
echo.
pause
