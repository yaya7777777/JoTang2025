@echo off
chcp 65001 > nul

set "SCRIPT_NAME=fix_markdown_images.py"
set "SCRIPT_PATH=%~dp0%SCRIPT_NAME%"

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python。请先安装Python并将其添加到系统PATH中。
    pause
    exit /b 1
)

REM 检查脚本文件是否存在
if not exist "%SCRIPT_PATH%" (
    echo 错误: 未找到脚本文件 %SCRIPT_NAME%
    pause
    exit /b 1
)

REM 创建日志文件
set "LOG_FILE=%~dp0markdown_image_fix_log.txt"

echo. > "%LOG_FILE%"
echo Markdown图片路径修复工具 >> "%LOG_FILE%"
echo 运行时间: %date% %time% >> "%LOG_FILE%"
echo --------------------------------------------- >> "%LOG_FILE%"

echo =============================================
echo Markdown图片路径修复工具
echo =============================================
echo 此工具将帮助修复所有Markdown文件中的图片引用路径问题。
echo 修复后，图片应该能在VSCode的侧边预览中正常显示。
echo 所有修改过的文件都会创建备份文件(.bak.时间戳)。
echo.
echo 开始修复...
echo =============================================
echo.

REM 运行Python脚本并将输出写入日志文件
python "%SCRIPT_PATH%" >> "%LOG_FILE%" 2>&1

REM 显示日志文件内容
type "%LOG_FILE%"

echo.
echo =============================================
echo 修复完成!
echo 详细日志已保存至: %LOG_FILE%
echo.
echo 注意事项:
echo 1. 所有修改过的Markdown文件都已创建备份
   备份文件格式为: 原文件名.bak.时间戳
echo 2. 如果需要恢复原状，可以使用备份文件覆盖原文件
echo 3. 修复后的图片引用使用了绝对路径，应该能在任何预览模式下正常显示

echo =============================================
pause