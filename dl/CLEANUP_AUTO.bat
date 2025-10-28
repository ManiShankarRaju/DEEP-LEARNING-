@echo off
title Auto Cleanup Project for GitHub
color 0C

echo.
echo ========================================================================
echo                   AUTO CLEANUP FOR GITHUB
echo ========================================================================
echo.
echo Deleting unnecessary files...
echo.

REM Delete duplicate restoration versions
if exist "ai_restoration_gui.py" del /f /q "ai_restoration_gui.py"
if exist "ai_restoration_model.py" del /f /q "ai_restoration_model.py"
if exist "working_dl_restoration.py" del /f /q "working_dl_restoration.py"
if exist "working_dl_gui.py" del /f /q "working_dl_gui.py"
if exist "working_crack_removal.py" del /f /q "working_crack_removal.py"
if exist "working_gui.py" del /f /q "working_gui.py"
if exist "simple_gui.py" del /f /q "simple_gui.py"
if exist "simple_run.py" del /f /q "simple_run.py"
if exist "simple_setup.py" del /f /q "simple_setup.py"
if exist "instant_ai_restore.py" del /f /q "instant_ai_restore.py"
if exist "instant_demo.py" del /f /q "instant_demo.py"
if exist "fast_start.py" del /f /q "fast_start.py"
if exist "quick_start.py" del /f /q "quick_start.py"
if exist "quick_enhance.py" del /f /q "quick_enhance.py"
if exist "smart_adaptive_restoration.py" del /f /q "smart_adaptive_restoration.py"
if exist "professional_restoration_ultimate.py" del /f /q "professional_restoration_ultimate.py"
if exist "professional_ultimate.py" del /f /q "professional_ultimate.py"
if exist "fixed_ultimate_restoration.py" del /f /q "fixed_ultimate_restoration.py"
if exist "hybrid_restoration.py" del /f /q "hybrid_restoration.py"
if exist "minimal_crack_removal.py" del /f /q "minimal_crack_removal.py"
if exist "ultimate_launcher.py" del /f /q "ultimate_launcher.py"
if exist "ultimate_face_restoration.py" del /f /q "ultimate_face_restoration.py"
if exist "AI_RESTORATION.bat" del /f /q "AI_RESTORATION.bat"
if exist "CRACK_REMOVAL.bat" del /f /q "CRACK_REMOVAL.bat"
if exist "INSTANT_AI_RESTORE.bat" del /f /q "INSTANT_AI_RESTORE.bat"
if exist "PROFESSIONAL_ULTIMATE.bat" del /f /q "PROFESSIONAL_ULTIMATE.bat"
if exist "SMART_RESTORATION.bat" del /f /q "SMART_RESTORATION.bat"
if exist "RESTORE_PHOTOS.bat" del /f /q "RESTORE_PHOTOS.bat"
if exist "instant.bat" del /f /q "instant.bat"
if exist "start.bat" del /f /q "start.bat"
if exist "crack_fix_cli.py" del /f /q "crack_fix_cli.py"
if exist "face_quality_cli.py" del /f /q "face_quality_cli.py"
if exist "USER_README.md" del /f /q "USER_README.md"
if exist "AI_RESTORATION_GUIDE.md" del /f /q "AI_RESTORATION_GUIDE.md"
if exist "FINAL_SOLUTION_GUIDE.md" del /f /q "FINAL_SOLUTION_GUIDE.md"
if exist "QUICK_START_GUIDE.md" del /f /q "QUICK_START_GUIDE.md"
if exist "train_ai_model.py" del /f /q "train_ai_model.py"
if exist "setup_models.py" del /f /q "setup_models.py"
if exist "test_smart_adaptive.py" del /f /q "test_smart_adaptive.py"
if exist "ai_photo_restoration.ipynb" del /f /q "ai_photo_restoration.ipynb"
if exist "complete_photo_restoration.ipynb" del /f /q "complete_photo_restoration.ipynb"
if exist "GUI.py" del /f /q "GUI.py"
if exist "predict.py" del /f /q "predict.py"
if exist "run.py" del /f /q "run.py"
if exist "cog.yaml" del /f /q "cog.yaml"
if exist "Dockerfile" del /f /q "Dockerfile"
if exist "kubernetes-pod.yml" del /f /q "kubernetes-pod.yml"
if exist "ansible.yaml" del /f /q "ansible.yaml"
if exist "download-weights" del /f /q "download-weights"
if exist "dl.code-workspace" del /f /q "dl.code-workspace"
if exist "show_results.py" del /f /q "show_results.py"
if exist "cleanup_for_github.py" del /f /q "cleanup_for_github.py"
if exist "cleanup_execute.py" del /f /q "cleanup_execute.py"
if exist "CODE_OF_CONDUCT.md" del /f /q "CODE_OF_CONDUCT.md"
if exist "SECURITY.md" del /f /q "SECURITY.md"
if exist "GITHUB_UPLOAD_GUIDE.md" del /f /q "GITHUB_UPLOAD_GUIDE.md"
if exist "CLEANUP_PROJECT.bat" del /f /q "CLEANUP_PROJECT.bat"

echo Deleting unnecessary directories...
echo.

if exist "__pycache__" rmdir /s /q "__pycache__"
if exist ".kiro" rmdir /s /q ".kiro"
if exist "Face_Detection" rmdir /s /q "Face_Detection"
if exist "Face_Enhancement" rmdir /s /q "Face_Enhancement"
if exist "Global" rmdir /s /q "Global"
if exist "imgs" rmdir /s /q "imgs"
if exist "models" rmdir /s /q "models"
if exist "temp_ai_output" rmdir /s /q "temp_ai_output"
if exist "training_data" rmdir /s /q "training_data"

echo.
echo ========================================================================
echo                       CLEANUP COMPLETE!
echo ========================================================================
echo.
echo Project is ready for GitHub!
echo.
echo Remaining files:
dir /b
echo.
echo Press any key to exit...
pause >nul

REM Delete this script itself
del /f /q "%~f0"
