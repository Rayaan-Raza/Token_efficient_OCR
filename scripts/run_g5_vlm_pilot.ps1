# G5 VLM pilot: n=300, K=2, OOF lgbm_strict + key baselines (BM25 is headline comparator)
# Same fairness protocol as G4: OOF scores, same pool/OCR/K/overview/prompt, cost logging.
Set-Location (Join-Path $PSScriptRoot "..")

$methods = @(
    "learned_lgbm_strict",
    "bm25_only",
    "bops_qa_fair_pool",
    "resize",
    "bops_fair_pool",
    "uniform"
)

$logDir = "outputs/logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMddTHHmmssZ"
$masterLog = Join-Path $logDir "g5_vlm_pilot_$stamp.log"

foreach ($m in $methods) {
    $msg = "$(Get-Date -Format o) START method=$m"
    Write-Host $msg
    Add-Content -Path $masterLog -Value $msg
    $methodLog = Join-Path $logDir "g5_vlm_${m}_$stamp.log"
    cmd /c "python scripts/run_vlm_eval.py --manifest Data/manifests/docvqa_300.jsonl --method $m --num-patches 2 --limit 300 --experiment-stage pilot --overwrite >> `"$methodLog`" 2>&1"
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

cmd /c "python scripts/bootstrap_pilot_stats.py --metric vlm --manifest docvqa_300 >> `"$masterLog`" 2>&1"
cmd /c "python scripts/merge_vlm_metrics.py >> `"$masterLog`" 2>&1"
Write-Host "G5 VLM pilot complete. Master log: $masterLog"
Add-Content -Path $masterLog -Value "$(Get-Date -Format o) ALL_DONE"
