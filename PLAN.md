# PLAN — LayaDecisionMcp

Servidor MCP local de **decisiones tipadas** de Laya (`choice` / `score` / `noul` con probabilidades calibradas y confianza) para OpenCode, implementado **100% en C# y PowerShell** (sin Python, sin Node). Publicación **Native AOT** (win-x64, .NET 10).

- Ruta: `D:\TodosProyectos\CSHARP\LayaDecisionMcp`
- Nombre verificado en GitHub (2026-09-24): `LayaMcpServer`, `LayaDecisionMcp`, `LayaDecision`, `opencode-laya` → **0 repos globales**; `user:cl2raul66` sin coincidencias → nombre libre para un futuro repo.

---

## 1. Decisión de arranque (tomada)

**Transporte: MCP local sobre stdio (`type: local`), gestionado por OpenCode.**

Razones (las más eficientes dadas las condiciones):
- OpenCode V2 levanta/recicla el proceso automáticamente → **cero gestión**: nada de servicios Windows, puertos ni procesos huérfanos.
- El exe **Native AOT** arranca prácticamente instantáneo (~10–30 ms) → el coste de spawn es despreciable.
- El cuello de botella real es la **inferencia** ONNX (~15–40 ms/petición), que se atiende en el mismo proceso; el protocolo MCP sobre stdio soporta peticiones concurrentes por *request ID*.
- Es el mismo patrón mental que ya usamos con `microsoft-learn`/`github`, pero local.
- **Upgrade documentado** (si algún día se necesita concurrencia multi-cliente): publicar el mismo host con `ModelContextProtocol.AspNetCore` (Streamable HTTP en Kestrel) y cambiar la config a `"type": "remote"`. Sin tocar el código de las tools.

Config prevista en `opencode.jsonc` (global):
```jsonc
"laya": {
  "type": "local",
  "command": "D:\\TodosProyectos\\CSHARP\\LayaDecisionMcp\\artifacts\\bin\\LayaDecisionMcp.exe"
}
```

## 2. Artefactos del modelo (sin Python)

Se descargan ya exportados vía **PowerShell** (`curl.exe -L`, sin `reproduce_int8.py` ni nada de Python):

**Fuente principal — `ti3x-m/laya-typed-decisions-onnx`** (base_model = `convaiinnovations/laya-typed-decisions`, el elegido):
- `onnx/model.onnx` + `onnx/model.onnx_data` (ONNX fp32 con datos externos)
- `tokenizer/tokenizer.json` + `tokenizer_config.json`
- `laya_config.json` (config del checkpoint: temperaturas, head budget…)
- `fixtures/reference.json` + `validation-fp32.json` → **fixtures doradas para testear el C# SIN Python** ⭐

**Fuente de cross-validación — `yehor-oleksiuk/laya-typed-decisions-onnx`** (CPU):
- `model_fp32.onnx`, `model_int8.onnx`, `tokenizer.json`
- Permite (a) verificar que dos exporters independientes del mismo checkpoint dan salidas equivalentes y (b) elegir `int8` como variante de rendimiento en CPU si cumple tolerancia.

URLs de descarga: `https://huggingface.co/<repo>/resolve/main/<fichero>` vía `curl.exe -L -o`.

## 3. Decodificación (port del formato ONNX documentado)

- Secuencia por pregunta: `[CLS] qtype question: instr [SEP] [MASK] opt0 [MASK] opt1 [SEP] state [SEP]` (`[MASK]` id 50284).
- Inputs del modelo: `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype` (0=choice, 1=score, 2=noul).
- Salida: softmax sobre los logits de los marcadores de las opciones de cada pregunta.
- Calibración: temperaturas por tipo desde `laya_config.json` (el model card oficial de `typed-decisions` cita `[1.0148, 1.0374, 1.0575]` con escalado por nº de opciones) + definición de **confianza** según el family (validar contra `reference.json`).
- Contexto máx 1024 tokens; truncación con flag `truncated: true` en la respuesta.
- Restricción family: `choice` con ≤ ~20 opciones (head compartido de 256 tokens).

## 4. Tokenizer (C# puro, dependencia cero, AOT-safe)

- Implementar `Laya.Tokenizer` en C# leyendo `tokenizer.json` de forma genérica (detecta WordPiece / BPE / Unigram según el `model.type` del JSON, con `added_tokens`). Son algoritmos cortos (~300–500 líneas), sin dependencias nativas, 100% compatibles con Native AOT.
- Validación: unit tests + fixtures que contengan `input_ids` y cruz contra el tokenizer de la segunda fuente.
- Alternativa descartada por ahora: `Microsoft.ML.OnnxRuntime.Extensions` (tokenizers de BERT) — solo se reintroduciría si el tipo real del tokenizer encaja perfecto y el benchmark lo justifica.

## 5. Estructura de la solución (.NET 10)

```
LayaDecisionMcp/
├─ PLAN.md
├─ tools/
│  └─ fetch-artifacts.ps1        # descarga/verifica artefactos (solo PowerShell)
├─ artifacts/                    # modelos ONNX + tokenizer + fixtures (descargado, ~4 GB)
├─ src/Laya.Tokenizer/           # carga tokenizer.json + encode (C# puro)
├─ src/Laya.Inference/           # ONNX Runtime: secuencia de marcadores, decode choice/score/noul,
│                                # softmax + calibración, truncation, mini-batching
├─ src/Laya.Mcp/                 # host MCP stdio (ModelContextProtocol SDK), herramientas
│                                # laya_evaluate / laya_info
└─ tests/Laya.Tests/             # xUnit: golden vs reference.json, cross-source int8/fp32,
                                 # schema JSON, truncation, multi-question
```

Interfaz del tool `laya_evaluate`:
- `state` (string, se trunca a 1024 tokens con flag) + `questions[]` `{ type: choice|score|noul, id, question, options[]|levels[]|criteria }`.
- Respuesta: `{ answers: [{ id, type, choice|score|noul, distribution[], confidence, truncated }] }`.
- `laya_info`: checkpoint cargado, contexto máx, latencia.

Dependencias NuGet: `Microsoft.ML.OnnxRuntime` (native via P/Invoke — compatible con AOT) + `ModelContextProtocol` (SDK MCP oficial C#, netstandard2.0 → net10). JSON con `JsonSerializerContext` (source generation). Sin reflexión en hot path.

## 6. Publicación Native AOT

```powershell
dotnet publish src/Laya.Mcp -c Release -r win-x64 -p:PublishAot=true -o artifacts\bin
```
- Prerequisitos verificados en la máquina: SDK .NET 10.0.4xx + VS 18 Community con herramientas C++ (VC.Tools.x86.x64) → listo.
- Validación sin herramientas externas: cliente MCP de prueba en C# (en `tests/`) + `opencode mcp list`.
- Si algo del SDK MCP fallara en AOT (riesgo bajo): transport* mínimo propio (JSON-RPC sobre stdio) con source-gen — 100% AOT-safe. *(ADR en F0)*

## 7. Fases

| Fase | Qué | Criterio de salida |
|---|---|---|
| **F0 · Verificación** | Inspeccionar `tokenizer.json` (tipo, vocab), `laya_config.json` (temperaturas/head), estructura de `fixtures/reference.json`; cargar `model.onnx` en ONNX Runtime C# y ejecutar 1 evaluación de humo; scaffold MCP stdio + `PublishAot` en win-x64 | ADR con: artefacto elegido, tokenizer confirmado, AOT validado |
| **F1 · Artefactos** | `tools/fetch-artifacts.ps1`: descargar ONNX + tokenizer + fixtures de las 2 fuentes, verificar tamaños/checksums | Equipo local reproducible con 1 comando |
| **F2 · Servidor C#** | Tokenizer → Inference (decode+calibración) → MCP stdio (`laya_evaluate`, `laya_info`) + tests golden (reference.json ±1e-4) y cross-source | `LayaDecisionMcp.exe` AOT responde tools en el cliente MCP de prueba |
| **F3 · Integración OpenCode** | `mcp` server `laya` (local), prompts de `gh-automation` (triage + puerta de seguridad), `gh-investigate` (diagnóstico CI + re-ranking), `ms-docs` (re-ranking), skill `laya-tools` | `opencode mcp list` → `✓ laya connected`; smoke `laya_evaluate` |
| **F4 · Ajuste** | Recalibrar temperaturas con datos propios (el model card admite que las del checkpoint no son fiables — issue #186), benchmark int8 vs fp32 en CPU, roadmap fine-tune `laya-multilingual` para estado en español | Uso en producción con umbrales conservadores |

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Export ONNX recién creados (2 días, pocas descargas) | Cross-validación entre 2 exporters independientes + fixtures `reference.json` antes de integrar; fallback `Mattepiu/laya-onnx` (más descargado) |
| `tokenizer.json` genérico → loader C# propio | Algoritmos estándar (WordPiece/BPE), unit tests + fixtures con `input_ids` |
| ONNX con datos externos (`model.onnx_data`) | `Microsoft.ML.OnnxRuntime` lo carga si ambos ficheros están juntos; verificar en F0 |
| SDK MCP en AOT | Transporte mínimo propio como fallback (ADR F0) |
| `typed-decisions` especialista e inglés | Servidor agnóstico al checkpoint: swap de artefactos sin tocar C#; umbrales conservadores; F4 fine-tune `multilingual` |
| Contexto 1024 tokens | Truncación con flag; skill `laya-tools` enseña a enviar resúmenes compactos |
| Espacio D: (25 GB libres) | Artefactos ~4–6 GB: suficiente; se informará en F1 |

## 9. Prerequisitos (confirmados presentes)

- .NET SDK 10.0.4xx · VS 18 Community (VC tools) · git 2.55 · PowerShell (Windows) · D: con 25.3 GB libres.
- OpenCode V2 (v2.0.16) ya configurado globalmente.

## 10. Entregables

1. Solución `LayaDecisionMcp` con publish AOT win-x64 (efecto binario en `artifacts/bin/`).
2. `tools/fetch-artifacts.ps1` reproducible.
3. Config OpenCode `mcp.servers.laya` + prompts de agentes `gh-*/ms-*` + skill `laya-tools`.
4. ADR de F0 con las decisiones técnicas y benchmarks.

## 11. Estado de implementación (2026-09-25) — hallazgos verificados empíricamente

Desviaciones respecto al plan original: las tools finales son `decision`/`temperatures`/`info`
(una decisión por llamada, no `laya_evaluate` multi-question); el binario AOT vive en
`publish\laya-mcp-win-x64\`; la solución usa `.slnx` (.NET 10).

### F0 (contrato ONNX) — cerrada
- Tokenizer **ByteLevel BPE GPT-2**: vocab 50 280, merges 50 009, normalizador NFC,
  `pre_tokenizer use_regex=true`, `add_prefix_space=false`; tabla byte→símbolo GPT-2,
  merges por rango mínimo. Specials: `[UNK]=50280, [CLS]=50281, [SEP]=50282, [PAD]=50283, [MASK]=50284`.
- Template: `[CLS] {qtype word} question: {instrucciones} [SEP] [MASK] opción … [SEP] {estado} [SEP]`
  con **espacio inicial obligatorio antes de cada opción** (`Encode("billing") ≠ Encode(" billing")`).
  `marker_pos` = posición absoluta del `[MASK]`; `marker_mask` = presencia; máx 3 opciones.
- **Temperaturas efectivas = `temperature_by_options` directas** (NO el array base `temperature`).
- Modelo **ONNX batch fijo = 1** (`dims=[1,-1]`); `reference.npz` = 3 ejecuciones batch-1 apiladas.
- ti3x: salidas `logits` + `act_probs` (ya normalizada, raw `[1,0]`); `act_probability = actProbs[0]`.
- Golden targets: department choice `[0.7876,0.127,0.0854]` conf 0.399 temp `choice:3-5`=1.7601518630981445;
  urgency score 1.4596 `[0.0963,0.3478,0.5559]` temp `score:3-5`=1.2514300346374512;
  churn noul 0.274 `[0.726,0.274]` temp `noul:2`=1.983399510383606; act 1.0 en los 3.

### F2 (servidor C#) — cerrada, tests 9/9
- `Laya.Tokenizer` (C# puro, `[GeneratedRegex]`, sin reflexión) + `Laya.Inference`
  (`LayaConfig`/`LayaSequence`/`LayaModel`) + `Laya.Mcp` (`LayaRuntime`/`LayaTools`/`Program`).
- `LayaRuntime`: tokenizer/config eager al arrancar, **ONNX Lazy** (1.6 GB, ~12 s en 1ª llamada);
  resolución `--model-dir` → `LAYA_MODEL_DIR` → `./artifacts` junto al exe → `%USERPROFILE%\.cache\laya\artifacts`.
- API real SDK MCP 2.2.0: `AddMcpServer().WithStdioServerTransport().WithTools<T>()` exige **clase de instancia**
  (no `static`); **no existe `McpProtocolException`** → validación con `ArgumentException` capturada y
  devuelta como JSON `{"error": …}` (`LayaToolError`) para que el LLM corrija; logging a stderr;
  `JsonSerializerOptions` manual (no `JsonSerializerOptions.Web`) para cero warnings de trim.
- Gating en `decision`: warnings si temp 1.0 por defecto, `actProbability < 0.6` (escalar) o `confidence < 0.5`.
- Publish AOT: `Laya.Mcp.exe` 14 MB + `onnxruntime.dll` 15.7 MB incluida; smoke stdio
  (`initialize → tools/list → tools/call`) idéntico en Debug y AOT.

### Cross-validation yehor — cerrada
- yehor exporta salidas `logits` + **`act_logits` (sin normalizar)** → `LayaModel.ExecuteRaw` aplica
  softmax estable y mantiene el contrato "act siempre normalizada" (`PickOutput` acepta el alias).
- `tokenizer.json` yehor **idéntico** al de ti3x (mismo SHA256); `laya_config.json` copiado de ti3x
  (mismo checkpoint base; el exportador ti3x reporta `validation-fp32.json: passed`, atol 1e-3).
- Tests parametrizados (`EndToEnd_*` × {ti3x, fp32, int8}, `Yehor_Logits_*` × {fp32, int8}):
  **fp32 reproduce los golden con la misma tolerancia que ti3x**; int8 mantiene argmax/decisiones
  (probs ±2e-2, score/conf ±5e-2). int8 viable para CPU si esa desviación es aceptable (F4).

### F3 (integración OpenCode) — hecha excepto skill
- `mcp.servers.laya` global (`type: local`, exe AOT + `--model-dir …\artifacts\ti3x`);
  `opencode mcp list` → `✓ laya connected`; tools `laya_decision`/`laya_temperatures`/`laya_info`
  verificadas de punta a punta vía OpenCode (números = golden).
- Agentes `gh-automation` (triaje+puerta), `gh-investigate` (reporta act/confidence, provisional si
  `act<0.6` o `conf<0.5`), `ms-docs` (desambiguación) actualizados con gating.
- **Pendiente**: skill `laya-tools`, `git init` del proyecto, y F4 (recalibración issue #186, benchmark
  int8 vs fp32 en CPU, fine-tune multilingüe).
- Caveatas vigentes: inglés-only, contexto ≤1024, máx 3 opciones, combos sin bucket → temp 1.0
  por defecto (warning en la respuesta).