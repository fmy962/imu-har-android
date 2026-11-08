# push_to_github.ps1
# One-click push the Android project to GitHub so GitHub Actions builds the APK.
#
# HOW TO USE:
#   1. Create an empty repo at https://github.com/new
#      (name e.g. imu-har-android, Private is fine, do NOT add README/gitignore)
#   2. Edit the $repo line below to your repo URL
#   3. Run this script:
#      powershell -ExecutionPolicy Bypass -File "C:\Users\FANGMINGYU\WorkBuddy\作业\imu_har\android\push_to_github.ps1"
#
# If Git asks for credentials, use your GitHub USERNAME and a Personal Access Token
# (NOT your password). Token needs "repo" scope.
# Create token at: https://github.com/settings/tokens

$ErrorActionPreference = "Stop"

$repo = "https://github.com/YOUR_USERNAME/imu-har-android.git"   # <-- EDIT THIS LINE
$dir  = "C:\Users\FANGMINGYU\WorkBuddy\作业\imu_har\android"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: git not found. Install Git for Windows from https://git-scm.com then re-run."
    exit 1
}

Set-Location $dir
Write-Host "Repo dir: $dir"

git remote remove origin 2>$null
git remote add origin $repo
git branch -M main

Write-Host "Pushing to $repo (first time GitHub will ask for user + token)..."
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "DONE. Go to your GitHub repo -> Actions tab -> 'Build Debug APK'."
    Write-Host "When it finishes, download the 'app-debug-apk' artifact (contains app-debug.apk)."
} else {
    Write-Host "PUSH FAILED. Check the repo URL and your GitHub login/token."
}
