@echo off
rem Build and run the api_http parser/router unit tests. From the REPO ROOT:
rem   api\tests\build-api-http-test.bat [outdir]
rem
rem Links the bundled jansson sources directly, so this needs nothing installed
rem and no miner objects: api_http.c has no miner symbols, which is the point.

setlocal
set "OUT=%~1"
rem Never default to the repo root: cl drops a dozen .obj files wherever it is told to.
if "%OUT%"=="" set "OUT=%TEMP%\ccminer-api-http-test"
if not exist "%OUT%" mkdir "%OUT%"
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29 >nul
if errorlevel 1 (echo vcvars failed & exit /b 1)

cl /nologo /W3 /Od /MT /D_CRT_SECURE_NO_WARNINGS /DWIN32 ^
   /I. /Icompat /Icompat\jansson ^
   /Fe"%OUT%\api_http_test.exe" /Fo"%OUT%\\" ^
   api\tests\api_http_test.c core\api\api_http.c ^
   compat\jansson\dump.c compat\jansson\error.c compat\jansson\hashtable.c ^
   compat\jansson\load.c compat\jansson\memory.c compat\jansson\pack_unpack.c ^
   compat\jansson\strbuffer.c compat\jansson\strconv.c compat\jansson\utf.c ^
   compat\jansson\value.c ^
   ws2_32.lib || exit /b 1

echo.
"%OUT%\api_http_test.exe"
