param(
  [Parameter(Mandatory)][string]$Log,
  [Parameter(Mandatory)][string]$Stop,
  [Parameter(Mandatory)][string]$Exclude,
  [string]$Roots = ''
)
$ErrorActionPreference = 'Stop'
$skip = (Resolve-Path -LiteralPath $Exclude).Path.TrimEnd('\')
$chain = @()
$d = Split-Path $skip -Parent
while ($d) { $chain += $d.TrimEnd('\'); $d = Split-Path $d -Parent }
$targets = @()
foreach ($a in $chain) {
  $targets += , @(($a + '\'), $false)
  Get-ChildItem -LiteralPath ($a + '\') -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $chain -notcontains $_.FullName.TrimEnd('\') -and $_.FullName.TrimEnd('\') -ne $skip } |
    ForEach-Object { $targets += , @($_.FullName, $true) }
}
foreach ($r in ($Roots -split ';' | Where-Object { $_ })) {
  if (-not (Test-Path -LiteralPath $r)) { continue }
  $full = (Resolve-Path -LiteralPath $r).Path.TrimEnd('\')
  if ($skip.StartsWith($full + '\', 'OrdinalIgnoreCase')) { throw "root $full holds the excluded folder $skip" }
  $targets += , @($full, $true)
}
$global:q = [Collections.Concurrent.ConcurrentQueue[string]]::new()
$i = 0
foreach ($t in $targets) {
  try {
    $w = New-Object IO.FileSystemWatcher $t[0]
    $w.IncludeSubdirectories = $t[1]
    $w.InternalBufferSize = 65536
    $w.NotifyFilter = [IO.NotifyFilters]'FileName, DirectoryName, LastWrite, Size, CreationTime'
    foreach ($ev in 'Created', 'Changed', 'Deleted', 'Renamed') {
      Register-ObjectEvent -InputObject $w -EventName $ev -SourceIdentifier "w$i$ev" -Action {
        $global:q.Enqueue("$(Get-Date -Format 'HH:mm:ss.fff') $($Event.SourceEventArgs.ChangeType) $($Event.SourceEventArgs.FullPath)")
      } | Out-Null
    }
    Register-ObjectEvent -InputObject $w -EventName Error -SourceIdentifier "w${i}Err" -Action {
      $global:q.Enqueue("$(Get-Date -Format 'HH:mm:ss.fff') OVERFLOW $($Event.SourceEventArgs.GetException().Message)")
    } | Out-Null
    $w.EnableRaisingEvents = $true
    Set-Variable -Name "keep$i" -Value $w -Scope Global
    $i++
  } catch { "$(Get-Date -Format 'HH:mm:ss.fff') UNWATCHED $($t[0]) $($_.Exception.Message)" | Add-Content -LiteralPath $Log }
}
"$(Get-Date -Format 'HH:mm:ss.fff') WATCHING $i folders, all but $skip" | Add-Content -LiteralPath $Log
function Drain {
  $line = $null
  $buf = [Collections.Generic.List[string]]::new()
  while ($global:q.TryDequeue([ref]$line)) { $buf.Add($line) }
  if ($buf.Count) { $buf | Add-Content -LiteralPath $Log }
}
while (-not (Test-Path -LiteralPath $Stop)) { Start-Sleep -Milliseconds 200; Drain }
Start-Sleep -Milliseconds 500
Drain
"$(Get-Date -Format 'HH:mm:ss.fff') END" | Add-Content -LiteralPath $Log
