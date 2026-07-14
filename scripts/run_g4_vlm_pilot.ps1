# G4 lean VLM pilot: n=100, K=2, OOF lgbm_strict + fair baselines
# Note: do not use ErrorActionPreference Stop — torch/HF write warnings to stderr.
Set-Location (Join-Path $PSScriptRoot "..")

$methods = @(
    "learned_lgbm_strict",
    "bops_qa_fair_pool",
    "resize",
    "bops_fair_pool",
    "bm25_only",
    "uniform"
)

$logDir = "outputs/logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMddTHHmmssZ"
$masterLog = Join-Path $logDir "g4_vlm_pilot_$stamp.log"

foreach ($m in $methods) {
    $msg = "$(Get-Date -Format o) START method=$m"
    Write-Host $msg
    Add-Content -Path $masterLog -Value $msg
    $methodLog = Join-Path $logDir "g4_vlm_${m}_$stamp.log"
    cmd /c "python scripts/run_vlm_eval.py --manifest Data/manifests/docvqa_100.jsonl --method $m --num-patches 2 --limit 100 --experiment-stage pilot --overwrite >> `"$methodLog`" 2>&1"
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $fail = "$(Get-Date -Format o) FAIL method=$m exit=$code"
        Write-Host $fail
        Add-Content -Path $masterLog -Value $fail
        Get-Content $methodLog -Tail 40 | Write-Host
        exit $code
    }
    $ok = "$(Get-Date -Format o) DONE method=$m"
    Write-Host $ok
    Add-Content -Path $masterLog -Value $ok
}

cmd /c "python scripts/merge_vlm_metrics.py >> `"$masterLog`" 2>&1"
Write-Host "G4 VLM pilot complete. Master log: $masterLog"
Add-Content -Path $masterLog -Value "$(Get-Date -Format o) ALL_DONE"
