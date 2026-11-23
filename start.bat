@echo off
rem local-Higgs (Windows) - double-click to start. Add --dry or -dry for a test without Kaggle.
set ARG=
if /i "%~1"=="--dry" set ARG=-dry
if /i "%~1"=="-dry" set ARG=-dry
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %ARG%
pause
