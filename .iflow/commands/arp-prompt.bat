@echo off
REM ARP Prompt Optimizer Command V1.0
REM =================================

REM Get parameters
set ARGS=%*

REM Check if empty parameters
if "%ARGS%"=="" (
    goto :show_help
)

REM Check special commands
if "%ARGS%"=="--help" goto :show_help
if "%ARGS%"=="-h" goto :show_help
if "%ARGS%"=="--version" goto :show_version
if "%ARGS%"=="-v" goto :show_version

REM Switch to project root directory
cd /d "%~dp0..\..\"

REM Execute ARP command
echo ARP Intelligent Prompt Optimizer V1.0
echo =====================================
echo.

REM Execute Python script
python .iflow\commands\arp-prompt %ARGS%

goto :end

:show_help
echo.
echo ARP Intelligent Prompt Optimizer V1.0
echo =====================================
echo.
echo Usage:
echo   /arp-prompt "Your prompt"                    - Direct optimization
echo   /arp-prompt --mode professional "prompt"     - Professional mode
echo   /arp-prompt --mode beginner "prompt"         - Beginner mode  
echo   /arp-prompt --interactive                    - Interactive session
echo   /arp-prompt --stats                          - User statistics
echo   /arp-prompt --export                         - Export user data
echo.
echo Optimization modes:
echo   standard      - Standard optimization (default)
echo   professional  - Professional direction, add technical details
echo   beginner      - Beginner friendly, easy to understand
echo   ai_format     - AI friendly format, structured prompts
echo   reoptimize    - Re-optimize, improve based on feedback
echo.
echo Examples:
echo   /arp-prompt "Help me write a Python function"
echo   /arp-prompt --mode professional "Explain machine learning"
echo   /arp-prompt --mode beginner "What is blockchain"
echo   /arp-prompt --interactive
echo   /arp-prompt --stats
echo.
echo Data storage: data\prompt_optimizer\
echo Privacy protection: All data stored locally only
echo.
goto :end

:show_version
echo.
echo ARP Intelligent Prompt Optimizer V1.0
echo Developer: iFlow Architecture Team
echo Release Date: 2025-11-17
echo.

:end
endlocal