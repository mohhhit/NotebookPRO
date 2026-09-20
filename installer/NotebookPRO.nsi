!include "MUI2.nsh"

Name "NotebookPRO"
OutFile "..\installer_output\NotebookPRO_Setup.exe"
InstallDir "$LOCALAPPDATA\NotebookPRO"
RequestExecutionLevel user ; No UAC prompt needed

!define MUI_ABORTWARNING

; Installer Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "NotebookPRO" SecCore
  SetOutPath "$INSTDIR"
  
  ; Extract all files from the staging directory
  File /r "..\installer_output\stage\*"
  
  ; Set Environment variable to isolate pip from global user packages
  System::Call 'Kernel32::SetEnvironmentVariable(t "PYTHONNOUSERSITE", t "1")'
  
  ; Show detailed log
  DetailPrint "Installing pip..."
  nsExec::ExecToLog '"$INSTDIR\python\python.exe" "$INSTDIR\python\get-pip.py" --no-warn-script-location'
  
  DetailPrint "Installing AI dependencies (this may take a few minutes)..."
  nsExec::ExecToLog '"$INSTDIR\python\python.exe" -m pip install -r "$INSTDIR\requirements.txt" --no-warn-script-location'
  
  ; Create Start Menu shortcut
  CreateShortcut "$SMPROGRAMS\NotebookPRO.lnk" "$INSTDIR\notebook_pro_app.exe"
  
  ; Create Uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  
  ; Add to Windows "Add/Remove Programs"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NotebookPRO" "DisplayName" "NotebookPRO"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NotebookPRO" "DisplayIcon" "$INSTDIR\notebook_pro_app.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NotebookPRO" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NotebookPRO" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\NotebookPRO.lnk"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NotebookPRO"
SectionEnd

