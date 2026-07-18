@echo off
rem Сборка FacadeStudio.exe (запускать на Windows из папки проекта)
chcp 65001 >nul

python -m pip install --upgrade pyinstaller pillow laspy numpy opencv-python-headless ezdxf lazrs
if errorlevel 1 goto err

python -m PyInstaller --noconfirm --onefile --windowed --name FacadeStudio ^
    --collect-all laspy --collect-all lazrs facade_gui.py
if errorlevel 1 goto err

echo.
echo Готово: dist\FacadeStudio.exe
pause
exit /b 0

:err
echo.
echo Сборка не удалась — пришлите текст ошибки выше.
pause
exit /b 1
