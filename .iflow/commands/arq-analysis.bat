@echo off
REM ARQ分析命令 V16
REM =============

REM 获取参数
set QUERY=%*
if "%QUERY%"=="" set QUERY=系统分析

REM 运行轻量版分析
python .iflow\commands\arq-analysis-lite-v16.py %QUERY%