# Build script for Windows (PowerShell)
param([switch]$Clean)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location $RepoRoot

Write-Host "=== 1. Building Native C++ Engine (SurfInspectNative.dll) ===" -ForegroundColor Cyan
if ($Clean -and (Test-Path native/build)) {
    Remove-Item -Recurse -Force native/build
}

if (-not (Test-Path "vcpkg/scripts/buildsystems/vcpkg.cmake")) {
    Write-Host "Using default CMake generator..." -ForegroundColor Yellow
    cmake -B native/build -S native -DCMAKE_BUILD_TYPE=Release
} else {
    $vcpkgToolchain = (Resolve-Path "vcpkg/scripts/buildsystems/vcpkg.cmake").Path.Replace('\', '/')
    $env:CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
    cmake -B native/build -S native -DCMAKE_BUILD_TYPE=Release
}

cmake --build native/build --config Release

Write-Host "=== 2. Copying Built DLL to Package Locations ===" -ForegroundColor Cyan
Copy-Item -Force native/build/Release/SurfInspectNative.dll src/svr_roughness/native/SurfInspectNative.dll
if (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python313\Lib\site-packages\svr_roughness\native") {
    Copy-Item -Force native/build/Release/SurfInspectNative.dll "$env:LOCALAPPDATA\Programs\Python\Python313\Lib\site-packages\svr_roughness\native\SurfInspectNative.dll"
}

Write-Host "=== 3. Running Unit Tests ===" -ForegroundColor Cyan
python -m pytest -v tests

Write-Host "=== 4. Packaging Release Artifacts (.whl & .tar.gz) ===" -ForegroundColor Cyan
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path "vcpkg/scripts/buildsystems/vcpkg.cmake") {
    $vcpkgToolchain = (Resolve-Path "vcpkg/scripts/buildsystems/vcpkg.cmake").Path.Replace('\', '/')
    $env:CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
}
python -m build --sdist --no-isolation
python -m build --wheel --no-isolation

Write-Host "`nSuccessfully built release packages in dist/!" -ForegroundColor Green
Get-ChildItem dist
