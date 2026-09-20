$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$flutterRoot = Join-Path $projectRoot 'notebook_pro_app'
$installerOutput = Join-Path $projectRoot 'installer_output'
$stageDir = Join-Path $installerOutput 'stage'

Write-Host '[1/5] Preparing Output Directory...'
New-Item $installerOutput -ItemType Directory -Force | Out-Null
Remove-Item (Join-Path $installerOutput '*') -Recurse -Force -ErrorAction SilentlyContinue
New-Item $stageDir -ItemType Directory -Force | Out-Null

Write-Host '[2/5] Downloading Portable Python Environment...'
$pythonDir = Join-Path $stageDir 'python'
New-Item $pythonDir -ItemType Directory -Force | Out-Null

$pythonZip = Join-Path $installerOutput 'python.zip'
Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile $pythonZip
Expand-Archive -Path $pythonZip -DestinationPath $pythonDir -Force
Remove-Item $pythonZip -Force

# Enable site-packages
$pthFile = Join-Path $pythonDir 'python311._pth'
(Get-Content $pthFile) -replace '#import site', 'import site' | Set-Content $pthFile

# Download get-pip.py
Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile (Join-Path $pythonDir 'get-pip.py')

Write-Host '[3/5] Copying Backend Scripts...'
$backendSource = Join-Path $projectRoot 'backend'
$backendDest = Join-Path $stageDir 'backend'
& robocopy $backendSource $backendDest /E /XD data __pycache__ .pytest_cache .git /XF *.pyc
if ($LASTEXITCODE -ge 8) { throw "Robocopy failed to copy backend scripts." }
Copy-Item -Path (Join-Path $projectRoot 'backend\requirements.txt') -Destination $stageDir -Force

Write-Host '[4/5] Building Flutter Windows release...'
$flutterRelease = Join-Path $flutterRoot 'build\windows\x64\runner\Release'
Push-Location $flutterRoot
try {
  flutter clean
  if ($LASTEXITCODE -ne 0) { throw 'Flutter clean failed.' }
  flutter pub get
  if ($LASTEXITCODE -ne 0) { throw 'Flutter pub get failed.' }
  flutter build windows --release
  if ($LASTEXITCODE -ne 0) { throw 'Flutter Windows release build failed.' }
} finally {
  Pop-Location
}

if (-not (Test-Path $flutterRelease)) { throw "Flutter release bundle not found: $flutterRelease" }
Copy-Item -Path "$flutterRelease\*" -Destination $stageDir -Recurse -Force

Write-Host '[5/5] Downloading NSIS and Compiling Setup.exe...'
$nsisZip = Join-Path $installerOutput 'nsis.zip'
$nsisDir = Join-Path $installerOutput 'nsis'
curl.exe -L 'https://downloads.sourceforge.net/project/nsis/NSIS%203/3.10/nsis-3.10.zip' -o $nsisZip
Expand-Archive -Path $nsisZip -DestinationPath $nsisDir -Force
$makensis = Join-Path $nsisDir 'nsis-3.10\makensis.exe'

if (-not (Test-Path $makensis)) { throw "makensis.exe not found at $makensis" }

Push-Location (Join-Path $projectRoot 'installer')
try {
  & $makensis NotebookPRO.nsi
  if ($LASTEXITCODE -ne 0) { throw 'NSIS build failed.' }
} finally {
  Pop-Location
}

Write-Host "Installer created: $(Join-Path $installerOutput 'NotebookPRO_Setup.exe')"
