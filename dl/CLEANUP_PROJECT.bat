@echo off
title Cleanup Project for GitHub
color 0C

echo.
echo  ========================================================================
echo                      PROJECT CLEANUP FOR GITHUB
echo  ========================================================================
echo.
echo  This will DELETE unnecessary files and keep only:
echo  - ultimate_photo_restoration.py (Main GUI)
echo  - ultimate_cli.py (Command line)
echo  - ULTIMATE_RESTORATION.bat (Launcher)
echo  - README.md, LICENSE, requirements.txt
echo  - test_images/ (sample images)
echo.
echo  ========================================================================
echo.
echo  WARNING: This will permanently delete 53 files and 9 directories!
echo.
pause

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

REM Delete old batch files
if exist "AI_RESTORATION.bat" del /f /q "AI_RESTORATION.bat"
if exist "CRACK_REMOVAL.bat" del /f /q "CRACK_REMOVAL.bat"
if exist "INSTANT_AI_RESTORE.bat" del /f /q "INSTANT_AI_RESTORE.bat"
if exist "PROFESSIONAL_ULTIMATE.bat" del /f /q "PROFESSIONAL_ULTIMATE.bat"
if exist "SMART_RESTORATION.bat" del /f /q "SMART_RESTORATION.bat"
if exist "RESTORE_PHOTOS.bat" del /f /q "RESTORE_PHOTOS.bat"
if exist "instant.bat" del /f /q "instant.bat"
if exist "start.bat" del /f /q "start.bat"

REM Delete CLI tools
if exist "crack_fix_cli.py" del /f /q "crack_fix_cli.py"
if exist "face_quality_cli.py" del /f /q "face_quality_cli.py"

REM Delete old guides
if exist "USER_README.md" del /f /q "USER_README.md"
if exist "AI_RESTORATION_GUIDE.md" del /f /q "AI_RESTORATION_GUIDE.md"
if exist "FINAL_SOLUTION_GUIDE.md" del /f /q "FINAL_SOLUTION_GUIDE.md"
if exist "QUICK_START_GUIDE.md" del /f /q "QUICK_START_GUIDE.md"

REM Delete training files
if exist "train_ai_model.py" del /f /q "train_ai_model.py"
if exist "setup_models.py" del /f /q "setup_models.py"
if exist "test_smart_adaptive.py" del /f /q "test_smart_adaptive.py"

REM Delete notebooks
if exist "ai_photo_restoration.ipynb" del /f /q "ai_photo_restoration.ipynb"
if exist "complete_photo_restoration.ipynb" del /f /q "complete_photo_restoration.ipynb"

REM Delete old GUI
if exist "GUI.py" del /f /q "GUI.py"

REM Delete deployment files
if exist "predict.py" del /f /q "predict.py"
if exist "run.py" del /f /q "run.py"
if exist "cog.yaml" del /f /q "cog.yaml"
if exist "Dockerfile" del /f /q "Dockerfile"
if exist "kubernetes-pod.yml" del /f /q "kubernetes-pod.yml"
if exist "ansible.yaml" del /f /q "ansible.yaml"
if exist "download-weights" del /f /q "download-weights"

REM Delete workspace files
if exist "dl.code-workspace" del /f /q "dl.code-workspace"

REM Delete helper scripts
if exist "show_results.py" del /f /q "show_results.py"
if exist "cleanup_for_github.py" del /f /q "cleanup_for_github.py"
if exist "cleanup_execute.py" del /f /q "cleanup_execute.py"

REM Delete optional docs
if exist "CODE_OF_CONDUCT.md" del /f /q "CODE_OF_CONDUCT.md"
if exist "SECURITY.md" del /f /q "SECURITY.md"

echo.
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
echo                         CLEANUP COMPLETE!
echo ========================================================================
echo.
echo Your project is now ready for GitHub with only essential files:
echo   - ultimate_photo_restoration.py (Main GUI)
echo   - ultimate_cli.py (Command line version)
echo   - ULTIMATE_RESTORATION.bat (Windows launcher)
echo   - README.md, LICENSE, requirements.txt
echo   - .gitignore
echo   - test_images/ (sample images)
echo   - output/ (output directory)
echo.
echo Next steps for GitHub:
echo   1. Review remaining files
echo   2. Update README.md with project description
echo   3. git init (if not already done)
echo   4. git add .
echo   5. git commit -m "Initial commit - Ultimate Photo Restoration"
echo   6. git push to GitHub
echo.
echo Your project path: %CD%
echo.
pause
