# fetch-artifacts.ps1 — Descarga los artefactos ONNX/tokenizer/fixtures de Laya (solo PowerShell)
# Uso:
#   .\tools\fetch-artifacts.ps1             -> todo
#   .\tools\fetch-artifacts.ps1 -SkipModels -> solo ficheros pequeños (tokenizer/config/fixtures)
#   .\tools\fetch-artifacts.ps1 -ModelsOnly -> solo modelos ONNX grandes
param(
  [switch]$SkipModels,
  [switch]$ModelsOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$base = Join-Path $root 'artifacts'
New-Item -ItemType Directory -Force -Path $base | Out-Null

function Get-HF {
  param([string]$repo, [string]$file, [string]$dest)
  $dir = Split-Path -Parent $dest
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  if (Test-Path $dest) {
    $mb = [math]::Round((Get-Item $dest).Length/1MB, 1)
    Write-Host "[cache] $dest ($mb MB)" -ForegroundColor DarkGray
    return
  }
  $url = "https://huggingface.co/$repo/resolve/main/$file"
  Write-Host "==> $url" -ForegroundColor Cyan
  curl.exe -L -C - --fail --retry 3 -o $dest $url
  if ($LASTEXITCODE -ne 0) { throw "Fallo al descargar: $url" }
  $mb = [math]::Round((Get-Item $dest).Length/1MB, 1)
  Write-Host "  OK $([System.IO.Path]::GetFileName($dest)) ($mb MB)" -ForegroundColor Green
}

$t = Join-Path $base 'ti3x'
$y = Join-Path $base 'yehor'

if (-not $ModelsOnly) {
  Write-Host "--- Ficheros pequenos: ti3x-m/laya-typed-decisions-onnx" -ForegroundColor Yellow
  Get-HF 'ti3x-m/laya-typed-decisions-onnx' 'tokenizer/tokenizer.json'   (Join-Path $t 'tokenizer.json')
  Get-HF 'ti3x-m/laya-typed-decisions-onnx' 'tokenizer_config.json'      (Join-Path $t 'tokenizer_config.json')
  Get-HF 'ti3x-m/laya-typed-decisions-onnx' 'laya_config.json'           (Join-Path $t 'laya_config.json')
  Get-HF 'ti3x-m/laya-typed-decisions-onnx' 'fixtures/reference.json'    (Join-Path $t 'fixtures/reference.json')
  Get-HF 'ti3x-m/laya-typed-decisions-onnx' 'validation-fp32.json'       (Join-Path $t 'fixtures/validation-fp32.json')
  Write-Host "--- Ficheros pequenos: yehor-oleksiuk/laya-typed-decisions-onnx" -ForegroundColor Yellow
  Get-HF 'yehor-oleksiuk/laya-typed-decisions-onnx' 'tokenizer.json'      (Join-Path $y 'tokenizer.json')
}

if (-not $SkipModels) {
  Write-Host "--- Modelos ONNX: ti3x-m (fp32, datos externos)" -ForegroundColor Yellow
  Get-HF 'ti3x-m/laya-typed-decisions-onnx' 'onnx/model.onnx'             (Join-Path $t 'model.onnx')
  Get-HF 'ti3x-m/laya-typed-decisions-onnx' 'onnx/model.onnx_data'        (Join-Path $t 'model.onnx_data')
  Write-Host "--- Modelos ONNX: yehor-oleksiuk (fp32 + int8 CPU)" -ForegroundColor Yellow
  Get-HF 'yehor-oleksiuk/laya-typed-decisions-onnx' 'model_fp32.onnx'     (Join-Path $y 'model_fp32.onnx')
  Get-HF 'yehor-oleksiuk/laya-typed-decisions-onnx' 'model_int8.onnx'     (Join-Path $y 'model_int8.onnx')
}

Write-Host ""
Write-Host "Artefactos en: $base"
Get-ChildItem -Recurse -File $base | Sort-Object Length -Descending |
  Select-Object @{n='MB';e={[math]::Round($_.Length/1MB,1)}}, FullName | Format-Table -AutoSize