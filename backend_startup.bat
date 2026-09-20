
@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM NotebookPRO backend + localtunnel launcher (Windows)
REM Usage examples:
REM   backend_startup.bat
REM   backend_startup.bat --reload
REM   backend_startup.bat --backend-only
REM   backend_startup.bat --tunnel-only --subdomain my-notebook-pro
REM   backend_startup.bat --tunnel-window
REM   backend_startup.bat --random-subdomain
REM   backend_startup.bat --restart-backend
REM   backend_startup.bat --project-root D:\projects_temp\NoteBookPRO_og

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Item -LiteralPath '%~f0').Directory.FullName"`) do set "SCRIPT_DIR=%%I"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "PROJECT_ROOT="
set "PROJECT_ROOT_CLI="
set "BACKEND_DIR="
set "PYTHON_EXE="
set "PORT=8010"
set "HEALTH_URL=http://127.0.0.1:%PORT%/api/health"

set "START_BACKEND=1"
set "START_TUNNEL=1"
set "USE_RELOAD=0"
set "DRY_RUN=0"
set "LT_SUBDOMAIN=note-book-pro"
set "TUNNEL_INLINE=1"
set "BACKEND_STARTED_BY_SCRIPT=0"
set "BACKEND_PID="
set "BACKEND_REUSED_EXTERNAL=0"
set "FORCE_RESTART_BACKEND=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--backend-only" (
	set "START_TUNNEL=0"
	shift
	goto parse_args
)
if /I "%~1"=="--tunnel-only" (
	set "START_BACKEND=0"
	shift
	goto parse_args
)
if /I "%~1"=="--reload" (
	set "USE_RELOAD=1"
	shift
	goto parse_args
)
if /I "%~1"=="--dry-run" (
	set "DRY_RUN=1"
	shift
	goto parse_args
)
if /I "%~1"=="--tunnel-window" (
	set "TUNNEL_INLINE=0"
	shift
	goto parse_args
)
if /I "%~1"=="--tunnel-inline" (
	set "TUNNEL_INLINE=1"
	shift
	goto parse_args
)
if /I "%~1"=="--subdomain" (
	if "%~2"=="" (
		echo [ERROR] --subdomain requires a value.
		exit /b 1
	)
	set "LT_SUBDOMAIN=%~2"
	shift
	shift
	goto parse_args
)
if /I "%~1"=="--random-subdomain" (
	set "LT_SUBDOMAIN="
	shift
	goto parse_args
)
if /I "%~1"=="--restart-backend" (
	set "FORCE_RESTART_BACKEND=1"
	shift
	goto parse_args
)
if /I "%~1"=="--project-root" (
	if "%~2"=="" (
		echo [ERROR] --project-root requires a value.
		exit /b 1
	)
	set "PROJECT_ROOT_CLI=%~2"
	shift
	shift
	goto parse_args
)

echo [WARN] Unknown argument: %~1
shift
goto parse_args

:args_done
if defined PROJECT_ROOT_CLI (
	set "PROJECT_ROOT=%PROJECT_ROOT_CLI%"
) else if defined NOTEBOOKPRO_ROOT (
	set "PROJECT_ROOT=%NOTEBOOKPRO_ROOT%"
) else (
	set "PROJECT_ROOT=%SCRIPT_DIR%"
	if not exist "%PROJECT_ROOT%\backend\main.py" if exist "%CD%\backend\main.py" set "PROJECT_ROOT=%CD%"
)

if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "BACKEND_DIR=%PROJECT_ROOT%\backend"
set "PYTHON_EXE=%PROJECT_ROOT%\virt\Scripts\python.exe"

if not exist "%BACKEND_DIR%\main.py" if exist "D:\projects_temp\NoteBookPRO_og\backend\main.py" (
	set "PROJECT_ROOT=D:\projects_temp\NoteBookPRO_og"
	set "BACKEND_DIR=!PROJECT_ROOT!\backend"
	set "PYTHON_EXE=!PROJECT_ROOT!\virt\Scripts\python.exe"
)

echo ========================================
echo NotebookPRO Startup
echo Project root : %PROJECT_ROOT%
echo Backend path : %BACKEND_DIR%
echo Python path  : %PYTHON_EXE%
echo Port         : %PORT%
if "%LT_SUBDOMAIN%"=="" (
	echo Tunnel mode  : random localtunnel subdomain
) else (
	echo Tunnel URL   : https://%LT_SUBDOMAIN%.loca.lt
)
echo ========================================

if not exist "%BACKEND_DIR%\main.py" (
	echo [ERROR] Backend not found at %BACKEND_DIR%
	echo         Run from the project folder, or pass --project-root "D:\projects_temp\NoteBookPRO_og"
	echo         You can also set NOTEBOOKPRO_ROOT as a persistent environment variable.
	exit /b 1
)

if "%START_BACKEND%"=="1" (
	if not exist "%PYTHON_EXE%" (
		echo [ERROR] Python executable not found at %PYTHON_EXE%
		echo         Run from the project folder, or pass --project-root "D:\projects_temp\NoteBookPRO_og"
		exit /b 1
	)
)

if "%START_BACKEND%"=="1" (
	call :check_backend
	if /I "!BACKEND_HEALTHY!"=="True" (
		set "NEED_BACKEND_TAKEOVER=0"
		if "%FORCE_RESTART_BACKEND%"=="1" set "NEED_BACKEND_TAKEOVER=1"
		if "%FORCE_RESTART_BACKEND%"=="0" if "%START_TUNNEL%"=="1" if "%TUNNEL_INLINE%"=="1" set "NEED_BACKEND_TAKEOVER=1"

		if "!NEED_BACKEND_TAKEOVER!"=="1" (
			if "%DRY_RUN%"=="1" (
				echo [DRY-RUN] Would stop existing backend on port %PORT% and restart it.
			) else (
				if "%FORCE_RESTART_BACKEND%"=="1" (
					echo [INFO] Restart mode enabled. Stopping existing backend on port %PORT%...
				) else (
					echo [INFO] Existing backend detected. Taking ownership so Ctrl+C can stop both backend and tunnel...
				)
				call :stop_backend_by_port
				if errorlevel 1 exit /b 1
				call :wait_backend_down
				if errorlevel 1 exit /b 1
				call :start_backend
				if errorlevel 1 exit /b 1
			)
		) else (
			echo [OK] Backend already running on %HEALTH_URL%
			set "BACKEND_REUSED_EXTERNAL=1"
		)
	) else (
		if "%DRY_RUN%"=="1" (
			echo [DRY-RUN] Would start backend on port %PORT%
		) else (
			call :start_backend
			if errorlevel 1 exit /b 1
		)
	)
)

if "%START_TUNNEL%"=="1" (
	if "%TUNNEL_INLINE%"=="1" if "%BACKEND_REUSED_EXTERNAL%"=="1" (
		echo [WARN] Backend was already running before this script.
		echo [WARN] Ctrl+C will stop localtunnel only.
		echo [WARN] Use --restart-backend if you want Ctrl+C to stop both backend and tunnel.
	)

	if "%DRY_RUN%"=="1" (
		call :print_tunnel_cmd
		exit /b 0
	)

	call :start_tunnel
	if errorlevel 1 exit /b 1
)

echo [DONE] Startup sequence complete.
exit /b 0

:start_backend
set "UVICORN_ARGS=main:app --host 0.0.0.0 --port %PORT% --no-access-log"
if "%USE_RELOAD%"=="1" set "UVICORN_ARGS=%UVICORN_ARGS% --reload"

echo [INFO] Starting backend process...
if "%START_TUNNEL%"=="1" if "%TUNNEL_INLINE%"=="1" (
	echo [INFO] Backend is attached to this console. Ctrl+C will stop backend and tunnel together.
	start "NotebookPRO Backend" /B /D "%BACKEND_DIR%" "%PYTHON_EXE%" -X utf8 -m uvicorn %UVICORN_ARGS%
) else (
	start "NotebookPRO Backend" /MIN /D "%BACKEND_DIR%" "%PYTHON_EXE%" -X utf8 -m uvicorn %UVICORN_ARGS%
)

set /a WAITED=0
:wait_backend
call :check_backend
if /I "!BACKEND_HEALTHY!"=="True" (
	echo [OK] Backend is healthy at %HEALTH_URL%
	set "BACKEND_STARTED_BY_SCRIPT=1"
	call :resolve_backend_pid_by_port
	if defined BACKEND_PID (
		echo [INFO] Tracking backend PID !BACKEND_PID! for shutdown cleanup.
	) else (
		echo [WARN] Backend started, but PID could not be resolved yet.
	)
	exit /b 0
)

set /a WAITED+=1
if !WAITED! GEQ 45 (
	echo [ERROR] Backend did not become healthy in 45 seconds.
	exit /b 1
)

timeout /t 1 >nul
goto wait_backend

:check_backend
set "BACKEND_HEALTHY=False"
set "HTTP_CODE="
for /f %%H in ('curl.exe -s -o nul -w "%%{http_code}" --connect-timeout 2 --max-time 2 "%HEALTH_URL%"') do set "HTTP_CODE=%%H"
if "%HTTP_CODE%"=="200" set "BACKEND_HEALTHY=True"
exit /b 0

:wait_backend_down
set /a WAIT_DOWN=0
:wait_backend_down_loop
call :check_backend
if /I "!BACKEND_HEALTHY!"=="False" exit /b 0

set /a WAIT_DOWN+=1
if !WAIT_DOWN! GEQ 20 (
	echo [ERROR] Existing backend did not stop in time on port %PORT%.
	exit /b 1
)

timeout /t 1 >nul
goto wait_backend_down_loop

:print_tunnel_cmd
if "%LT_SUBDOMAIN%"=="" (
	echo [DRY-RUN] Would run: npx --yes localtunnel --port %PORT%
) else (
	echo [DRY-RUN] Would run: npx --yes localtunnel --port %PORT% --subdomain %LT_SUBDOMAIN%
)
exit /b 0

:start_tunnel
where npx >nul 2>&1
if errorlevel 1 (
	echo [ERROR] npx was not found. Install Node.js with npx support, then rerun this script.
	echo         Install Node.js, then rerun this script.
	exit /b 1
)

if "%TUNNEL_INLINE%"=="0" (
	if "%LT_SUBDOMAIN%"=="" (
		echo [INFO] Starting localtunnel on port %PORT% in a new terminal window...
		start "NotebookPRO LocalTunnel" cmd /k "npx --yes localtunnel --port %PORT%"
	) else (
		echo [INFO] Starting localtunnel on port %PORT% with subdomain %LT_SUBDOMAIN% in a new terminal window...
		start "NotebookPRO LocalTunnel" cmd /k "npx --yes localtunnel --port %PORT% --subdomain %LT_SUBDOMAIN%"
	)

	echo [OK] Localtunnel opened in a new terminal window.
	echo      Copy the https URL from that window into your client config.
	exit /b 0
)

echo [INFO] Starting localtunnel in this terminal...
echo [INFO] Wait for the "your url is:" line below.
if "%BACKEND_STARTED_BY_SCRIPT%"=="1" if defined BACKEND_PID (
	echo [INFO] Ctrl+C cleanup is enabled for backend PID !BACKEND_PID!.
	call :run_tunnel_with_backend_cleanup
) else (
	if "%LT_SUBDOMAIN%"=="" (
		npx --yes localtunnel --port %PORT%
	) else (
		npx --yes localtunnel --port %PORT% --subdomain %LT_SUBDOMAIN%
	)
)
set "LT_EXIT=%ERRORLEVEL%"

if "%BACKEND_STARTED_BY_SCRIPT%"=="1" if not defined BACKEND_PID (
	call :stop_backend_by_port
)

exit /b %LT_EXIT%

:stop_backend_by_port
call :resolve_backend_pid_by_port
if not defined BACKEND_PID (
	echo [WARN] Could not determine backend PID on port %PORT% for cleanup.
	exit /b 1
)

taskkill /F /T /PID !BACKEND_PID! >nul 2>&1
if errorlevel 1 (
	echo [WARN] Backend PID !BACKEND_PID! could not be stopped automatically.
	exit /b 1
)

echo [OK] Backend stopped (PID !BACKEND_PID!).
exit /b 0

:resolve_backend_pid_by_port
set "BACKEND_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /C:":%PORT%" ^| findstr /C:"LISTENING"') do (
	set "BACKEND_PID=%%P"
	goto resolve_backend_pid_done
)

:resolve_backend_pid_done
exit /b 0

:run_tunnel_with_backend_cleanup
if "%LT_SUBDOMAIN%"=="" (
	powershell -NoProfile -Command "$pidToKill=[int]$env:BACKEND_PID; try { & npx --yes localtunnel --port %PORT%; exit $LASTEXITCODE } finally { if ($pidToKill -gt 0) { & taskkill /F /T /PID $pidToKill } }"
) else (
	powershell -NoProfile -Command "$pidToKill=[int]$env:BACKEND_PID; try { & npx --yes localtunnel --port %PORT% --subdomain %LT_SUBDOMAIN%; exit $LASTEXITCODE } finally { if ($pidToKill -gt 0) { & taskkill /F /T /PID $pidToKill } }"
)
exit /b %ERRORLEVEL%