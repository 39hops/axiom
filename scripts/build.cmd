@echo off
rem Usage: scripts\build.cmd [test|bench|rel]
setlocal
set VSDEV="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
set CMAKE="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set CTEST="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
set NINJA="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
call %VSDEV% -arch=x64 -no_logo 2>nul
cd /d "%~dp0.."

if "%1"=="rel" (
  %CMAKE% -S . -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=%NINJA% && %CMAKE% --build build-rel
  exit /b %errorlevel%
)

if not exist build (
  %CMAKE% -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=%NINJA% || exit /b 1
)
%CMAKE% --build build || exit /b 1

if "%1"=="test" (
  %CTEST% --test-dir build --output-on-failure
  exit /b %errorlevel%
)
