@echo off
echo Adding Python to system PATH...

:: Add Python paths to system PATH
setx PATH "%PATH%;C:\Python313\;C:\Python313\Scripts\" /M

echo Python paths have been added to system PATH.
echo Please close and reopen your IDE for the changes to take effect.
pause 