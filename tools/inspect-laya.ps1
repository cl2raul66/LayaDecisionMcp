# inspect-laya.ps1 — Verificación F0/contrato ONNX (pwsh 7+ requerido para System.Text.Json)
# Decodifica tokenizer.json, reference.npz y descubre la fórmula de temperatura vs reference.json
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$t = Join-Path $root 'artifacts\ti3x'

function Read-Npy($path) {
  $b = [IO.File]::ReadAllBytes($path)
  $ver = $b[6]
  if ($ver -eq 1) { $hl = [BitConverter]::ToUInt16($b, 8) } else { $hl = [BitConverter]::ToUInt32($b, 8) }
  $desc = [Text.Encoding]::ASCII.GetString($b, 10, $hl)
  $shapeM = [regex]::Match($desc, "'shape'\s*:\s*\(([^)]*)\)")
  $shape = @(($shapeM.Groups[1].Value -split ',' | Where-Object { $_.Trim() -ne '' } | ForEach-Object { [int]$_.Trim() }))
  $dtype = [regex]::Match($desc, "'descr'\s*:\s*'([^']+)'").Groups[1].Value
  $off = 10 + $hl
  $count = 1; foreach ($s in $shape) { $count *= $s }
  $vals = @()
  switch ($dtype) {
    '<i8' { for ($i = 0; $i -lt $count; $i++) { $vals += [BitConverter]::ToInt64($b, $off + $i * 8) } }
    '<f4' { for ($i = 0; $i -lt $count; $i++) { $vals += [BitConverter]::ToSingle($b, $off + $i * 4) } }
    '|b1' { for ($i = 0; $i -lt $count; $i++) { $vals += [bool]$b[$off + $i] } }
    default { throw "dtype $dtype no soportado" }
  }
  return @{ shape = $shape; dtype = $dtype; vals = $vals }
}
function Get-NpyRow($npy, $row, $cols) {
  $start = $row * $cols
  return $npy.vals[$start..($start + $cols - 1)]
}

"========== TOKENIZER (ti3x) =========="
$doc = [System.Text.Json.JsonDocument]::Parse((Get-Content "$t\tokenizer.json" -Raw))
$r = $doc.RootElement
$model = $r.GetProperty('model')
"model.type      : $($model.GetProperty('type').GetString())"
$vocab = $model.GetProperty('vocab')
"vocab size      : $(@($vocab.EnumerateObject()).Count)"
"merges          : $($model.GetProperty('merges').GetArrayLength())"
$unkV = $null; foreach($p in $model.EnumerateObject()){ if($p.Name -eq 'unk_token'){ $unkV = $p.Value.GetString() } }
if ($null -ne $unkV) { "unk_token       : '$unkV'" }
"pre_tokenizer   : $($r.GetProperty('pre_tokenizer').GetRawText())"
"post_processor  : $($r.GetProperty('post_processor').GetRawText())"
"normalizer      : $($r.GetProperty('normalizer').GetRawText())"
"decoder         : $($r.GetProperty('decoder').GetRawText())"
"added_tokens    : $($r.GetProperty('added_tokens').GetArrayLength())"
foreach($e in $r.GetProperty('added_tokens').EnumerateArray()){
  $id = $e.GetProperty('id').GetInt32(); $c = $e.GetProperty('content').GetString()
  $sp = $e.GetProperty('special').GetBoolean()
  "  {0,6}  '{1}'  special={2}" -f $id, $c, $sp
}
# reverse map id->token
$rev = @{}
foreach($p in $vocab.EnumerateObject()){ $rev[[int]$p.Value.GetInt32()] = $p.Name }
foreach($e in $r.GetProperty('added_tokens').EnumerateArray()){ $rev[$e.GetProperty('id').GetInt32()] = $e.GetProperty('content').GetString() }
$doc.Dispose()

"========== reference.npz =========="
$refDir = "$t\fixtures\ref"
if (-not (Test-Path "$refDir\input_ids.npy")) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [System.IO.Compression.ZipFile]::ExtractToDirectory((Resolve-Path "$t\fixtures\reference.npz"), $refDir)
}
$inp = Read-Npy "$refDir\input_ids.npy"
$att = Read-Npy "$refDir\attention_mask.npy"
$mpos = Read-Npy "$refDir\marker_pos.npy"
$mmask = Read-Npy "$refDir\marker_mask.npy"
$qt = Read-Npy "$refDir\qtype.npy"
$lg = Read-Npy "$refDir\logits.npy"
$ap = Read-Npy "$refDir\act_probs.npy"

"qtype           : $(($qt.vals | ForEach-Object { [int]$_ }) -join ', ')"
"marker_pos      :"; for($i=0;$i -lt 3;$i++){ "  row$i : $(Get-NpyRow $mpos $i 3 | ForEach-Object {[int]$_}) -join ',' | mask $(Get-NpyRow $mmask $i 3 | ForEach-Object {[int]$_})" }
"logits          :"; for($i=0;$i -lt 3;$i++){ "  row$i : $((Get-NpyRow $lg $i 3 | ForEach-Object {[math]::Round([double]$_,5)}) -join ', ')" }
"act_probs       :"; for($i=0;$i -lt 3;$i++){ "  row$i : $((Get-NpyRow $ap $i 2 | ForEach-Object {[math]::Round([double]$_,5)}) -join ', ')" }
""
"--- input_ids decodificados (template) ---"
for($i=0;$i -lt 3;$i++){
  $row = Get-NpyRow $inp $i 72
  $toks = @(); $prev=-1
  for($j=0;$j -lt 72;$j++){
    $id=[int]$row[$j]
    if($id -eq $prev){ continue } # dedupe consecutive (puede haber subword split; mantenemos)
    $prev=$id
    if($rev.ContainsKey($id)){ $toks += $rev[$id] } else { $toks += "<#$id>" }
  }
  "  Q$i : $($toks -join '|')"
}

"========== EXPERIMENTO TEMPERATURA =========="
function Soft { param($l,$t) $s=@(); foreach($x in $l){$s += [double]$x/$t }; $m=($s|Measure-Object -Maximum).Maximum; $e=@();$sum=0; foreach($x in $s){$e+=[math]::Exp($x-$m);$sum+=$e[-1]}; $e|ForEach-Object{$_/$sum} }
# Urgency: qtype=1 (score), 3 niveles, logits row2? (orden preguntas: department=0, urgency=1, churn=2)
"REFERENCE probs: dep=[0.7876,0.0854,0.127] urgency=[0.0963,0.3478,0.5559] noul=0.274"
foreach($case in @(
  @{n='urgency';row=1;q='1';probs=@(0.0963,0.3478,0.5559);temps=@('1.0374*1.2514','1.0374','1.2514','1')},
  @{n='department';row=0;q='0';probs=@(0.7876,0.0854,0.127);temps=@('1.0148*1.7602','1.0148','1.7602','1')},
  @{n='churn';row=2;q='2';probs=@(0.726,0.274);temps=@('1.0575*1.9834','1.0575','1.9834','1')}
)){
  $l = Get-NpyRow $lg $case.row 3
  if($case.n -eq 'churn'){ $l = @($l[0],$l[1]) }
  "$($case.n) raw logits: $($l | ForEach-Object{[math]::Round([double]$_,4)}) ref: $($case.probs -join ',')"
  foreach($tp in $case.temps){
    $t = $tp -replace '\*','*'; $tv = Invoke-Expression $t
    $sm = Soft $l $tv
    $s = ($sm | ForEach-Object {[math]::Round($_,4)}) -join ','
    $err = 0; for($k=0;$k -lt $case.probs.Count;$k++){ $err = [math]::Max($err,[math]::Abs($sm[$k]-$case.probs[$k])) }
    "  temp=$tp -> softmax=[$s]  maxerr=$([math]::Round($err,4))"
  }
}
# act_probability: softmax/softmax de act_probs
"--- act_probs softmax (filas) ---"
for($i=0;$i -lt 3;$i++){ $l=Get-NpyRow $ap $i 2; $s=Soft $l 1; "  row$i : raw [$($l|%{[math]::Round([double]$_,4)})] softmax [$($s|%{[math]::Round($_,4)})]" }
# confidence check: 1 - H/Hmax de department
$p=@(0.7876,0.0854,0.127); $H=0; foreach($x in $p){ if($x -gt 0){$H -= $x*[math]::Log($x)} }
$conf = 1 - $H/[math]::Log(3)
"confidence(department) calculado=$([math]::Round($conf,4)) vs reference 0.399"