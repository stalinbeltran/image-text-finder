# P2 — Confirmar el no-sobreajuste con N semillas (protocolo.md §9).
#
# Cuatro runs IDÉNTICOS a `dirty-20-lambda_pos_1-0005` salvo la semilla de D
# (init de pesos y shuffle). Ese run entra como la 5ª semilla: por eso las
# recetas `p2-seed{2..5}` se generaron copiando su `config.json`, no a mano.
#
# SECUENCIAL A PROPÓSITO. En CPU el límite de workers es 1: torch ya usa todos
# los núcleos y cada run carga su PatchDataset entero en RAM. Lanzar los cuatro
# a la vez no acelera nada y se queda sin memoria (CLAUDE.md).
#
# Coste medido: ~1.150 s/época × 5 épocas × 4 runs ≈ 6,4 h.
#
# Uso:  .\scripts\p2-seeds.ps1
# Los logs quedan en runs\<name>\train.log; el progreso real está en
# runs\<name>\metrics.jsonl (una línea por época) y en status.json.

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "no existe $python" }

foreach ($seed in 2, 3, 4, 5) {
    $name = "p2-noover-seed$seed"

    # Un run ya hecho no se repite y NO se pisa: `RunStore.create` reserva el
    # nombre y falla si existe. Saltarlo aquí es lo que hace el script
    # reanudable — si la noche se corta en el tercero, se relanza tal cual.
    if (Test-Path (Join-Path $repo "runs\$name")) {
        Write-Host "[$name] ya existe, lo salto"
        continue
    }

    Write-Host "[$name] arrancando $(Get-Date -Format 'HH:mm:ss')"
    & $python -m itf.training.cli `
        --name $name `
        --patch-dataset dirty-20 `
        --network cnn-20-border `
        --recipe "p2-seed$seed" `
        --device cpu
    if ($LASTEXITCODE -ne 0) { throw "[$name] falló con código $LASTEXITCODE" }
    Write-Host "[$name] terminado $(Get-Date -Format 'HH:mm:ss')"
}

Write-Host ""
Write-Host "Los cuatro runs están. Analiza con:"
Write-Host "  .\.venv\Scripts\python.exe scripts\p2_analyze.py"
