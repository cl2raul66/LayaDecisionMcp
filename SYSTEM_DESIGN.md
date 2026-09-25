# SYSTEM_DESIGN — LayaDecisionMcp

Diseño completo del servidor MCP de decisiones tipadas Laya. Cada sección incluye la **decisión**, la
**justificación** (por qué es la mejor opción dado el contexto) y la **evidencia** (medición o verificación
empírica que la respalda). Si cambias el código, mantén este documento coherente (ver `CONTRIBUTING.md`).

Contexto: OpenCode V2 sobre Windows, máquina de desarrollo con CPU y ~8 GB de RAM, sin GPU.
Restricción autoimpuesta del proyecto: **solo C#/.NET y PowerShell**; nada de Python/Node en el flujo
de build y test.

---

## 1. Arquitectura por capas

```
Agente (OpenCode, cualquier cliente MCP)
   │  JSON-RPC 2.0 newline-delimited sobre stdio
   ▼
src/Laya.Mcp        Host genérico (Microsoft.Extensions.Hosting) + ModelContextProtocol SDK 2.2.0
   │                Tools: decision / temperatures / info; errores de validación como JSON
   ▼
src/Laya.Inference  LayaConfig (temperaturas) · LayaSequence (plantilla+marcadores) · LayaModel (ONNX)
   │                5 inputs -> 2 outputs; decode choice/score/noul + confidence + act
   ▼
src/Laya.Tokenizer  ByteLevel BPE GPT-2 puro en C# (tokenizer.json genérico)
   │
   ▼
Microsoft.ML.OnnxRuntime (nativo, P/Invoke)  →  ModernBERT 421M (checkpoint typed-decisions)
```

El binario publicado es **Native AOT win-x64**: `Laya.Mcp.exe` (14 MB) + `onnxruntime.dll` (15,7 MB),
sin instalación de .NET requerida en el equipo destino.

---

## 2. El modelo y por qué este despliegue

**Qué es Laya:** un clasificador *System-1* (ModernBERT, 421 M parámetros) entrenado para 4 flujos de
decisión tipada (routing, extraction, matching, boolean/*noul*). No redacta; devuelve una distribución
sobre opciones dadas. Se consume como checkpoint ONNX pre-exportado.

**Por qué ONNX Runtime y no PyTorch/llama.cpp:**
- El checkpoint ya existe exportado a ONNX por dos autores independientes (ti3x-m, yehor-oleksiuk);
  usar exportadores existentes elimina Python del pipeline por completo.
- `Microsoft.ML.OnnxRuntime` es la única dependencia nativa y es compatible con AOT (P/Invoke estable).
- **Evidencia**: dos exportadores independientes producen los mismos logits (validación cruzada fp32
  dentro de tolerancia 1e-3/1e-4; ver §7), lo que confirma que el despliegue no depende de caprichos de
  un solo exportador.

**Por qué Native AOT:** OpenCode spawnea el servidor por sesión; el exe AOT arranca en milisegundos y
no requiere runtime .NET instalado. El coste dominante nunca es el arranque sino la carga del modelo
(1,6 GB de pesos), que es inevitable en cualquier tecnología. AOT también impone disciplina
(source-gen, sin reflexión) que mantiene el hot path limpio.

**Sobre el consumo de RAM (pregunta frecuente):** el exe son 14 MB; el working set de ~1,7-2 GB viene
de los **pesos del modelo** (421 M × fp32 ≈ 1,6 GB), que ONNX Runtime mapea al crear la sesión. AOT
solo ahorra el overhead del runtime gestionado (~20-50 MB). La palanca de memoria real es el modelo
int8 (482 MB) — pero ver §7: en la CPU objetivo infiere 3× más lento, así que int8 queda como opción
documentada y fp32 como default. La carga es **perezosa** (`Lazy<LayaModel>`): quien no llama a
`decision` no paga RAM ni tiempo.

---

## 3. Tokenizer (src/Laya.Tokenizer)

**Decisión:** implementación propia de ByteLevel BPE leyendo `tokenizer.json` de forma genérica.

**Contrato verificado (fixtures del exportador):**
- Modelo `BPE` GPT-2: vocab 50 280, 50 009 merges, normalizador **NFC**, pre-tokenizer con
  `use_regex=true` (split GPT-2) y `add_prefix_space=false`.
- Tabla byte→símbolo GPT-2 (bytes 0x00-0xFF mapeados a caracteres Unicode seguros, p. ej. espacio→`Ġ`).
- Merges aplicados por **rango mínimo** (par de mayor prioridad primero), clásico BPE.
- Tokens especiales añadidos: `[UNK]=50280, [CLS]=50281, [SEP]=50282, [PAD]=50283, [MASK]=50284`.

**Justificación:** el tokenizer oficial vive en `tokenizers` (Rust/HF) o `Microsoft.ML.Tokenizers`
(paridad parcial de BPE byte-level en .NET 10, sin garantía de cubrir este tokenizer.json concreto).
Un loader propio son ~200 líneas, sin dependencias, 100 % AOT, y queda **probado bit a bit** contra
los `input_ids` del fixture (`Tokenizer_Reproduces_Reference_InputIds`). Este es exactamente el tipo
de componente donde un fixture dorado permite reimplementar sin riesgo: la referencia *es* el test.

**Detalle crítico:** ByteLevel hace `"billing"` ≠ `" billing"` (el espacio se codifica dentro del
token: `Ġbilling`). La plantilla de opciones de Laya lo exige (§4), y sin ese espacio los logits no
coinciden con la referencia.

## 4. Secuencia de entrada y marcadores (LayaSequence)

**Formato verificado** (reconstruido de `reference.npz`):

```
[CLS] {qtype} question: {instructions} [SEP] [MASK] opción0 [MASK] opción1 [MASK] opción2 [SEP] {state} [SEP]
```

- `qtype` ∈ {choice, score, noul} como palabra; también se envía como input entero `qtype` (0/1/2).
- Cada opción lleva su **[MASK]** delante y un **espacio inicial** (`" " + opción`).
- `marker_pos`: posición absoluta de cada `[MASK]` (3 slots).
- `marker_mask`: presencia por slot (las posiciones ausentes reciben logit -10000 del exportador y se
  ignoran vía la máscara).
- Contexto máximo 1024 tokens (el builder lanza si se excede).

**Por qué máx 3 opciones:** la cabeza de marcadores es compartida con `head_max_len = 256`; el
checkpoint se calibró para hasta 3 marcadores por flujo en estos exports. Más allá, la calibración
deja de aplicar (no es una limitación nuestra: es el contrato del modelo).

## 5. Salidas y decode (LayaModel)

Inputs ONNX: `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`. **Batch fijo = 1**
(los exports declaran `dims=[1,-1]`; la referencia apila 3 ejecuciones batch-1).

Salidas por exportador:

| Exportador | Salida principal | Cabeza act | Notas |
|---|---|---|---|
| ti3x-m | `logits` | `act_probs` (softmax ya aplicada, `[1,0]` en los golden) | fuente principal |
| yehor-oleksiuk | `logits` | **`act_logits`** (sin normalizar) | validación cruzada |

**Decisión:** `ExecuteRaw` garantiza el contrato interno *"act siempre normalizada"*: si la salida se
llama `act_logits`, aplica softmax estable dentro. **Justificación:** un solo punto de verdad evita que
el decode dependa del exportador; el alias se resuelve una vez en el constructor (`PickOutput`), no por
llamada. **Evidencia:** test `Yehor_Logits_Match_Reference` compara contra la misma referencia.

**Decode** (sobre logits de marcadores válidos, temperatura `T`):
`pᵢ = softmax(logitᵢ / T)` con max-subtraction; `choice` = argmax; `score` = Σ pᵢ·i (niveles 0..N-1);
`noul` = p de la última opción ("true"). `confidence = 1 − H/ln(N)`. **Verificado** al decimal contra
`reference.json` (p. ej. urgency = 1,4596).

## 6. Calibración de temperatura (LayaConfig)

**Hallazgo empírico de F0:** las temperaturas efectivas son las de **`temperature_by_options`**
(bucket por tipo y nº de opciones: `choice:2`, `choice:3-5`, `noul:2`…), **no** el array base
`temperature` del model card ([1.0148, 1.0374, 1.0575]) ni productos de ambos. Se verificó
reproduciendo las probabilidades del fixture exactamente con el valor del bucket.

Defaults: combinación sin bucket → `T = 1.0`, y la tool `decision` lo anuncia en `warnings`
(la calibración del checkpoint tiene cobertura parcial; la propia documentación de Laya reconoce
recalibración pendiente — issue #186).

**Regla derivada:** `confidence` no está calibrada; en los agentes solo se usa como aviso, y el gating
firme es `actProbability` (la cabeza act/escalate entrenada para eso).

## 7. Rendimiento medido (CPU, win-x64, AOT)

Benchmark sobre las 3 decisiones golden por sesión de proceso (1ª llamada incluye carga del modelo):

| Modelo | Carga (1ª llamada) | Inferencia caliente | Resultados golden |
|---|---|---|---|
| ti3x fp32 (external data) | 13-32 s | 0,36-0,92 s | exactos (tolerancia 1e-4) |
| yehor fp32 | ~27 s | ~0,97 s | idénticos |
| yehor int8 | ~19 s | 2,8 s | argmax idéntico, probs ±2e-2 |

**Justificación de fp32 como default:** int8 ahorra RAM (482 MB vs 1,6 GB) pero en esta CPU infiere ~3×
más lento: con batch=1 y secuencias de ~70 tokens, el coste de cuantizar/decuantizar por capa supera lo
que ahorran los kernels int8 frente a oneDNN/MKL fp32. Conclusión generalizable: **int8 solo compensa
por memoria, no por velocidad**, en este perfil de uso.

**Optimización descartada con evidencia:** forzar `ExecutionMode.ORT_SEQUENTIAL` — es el modo por
defecto de ORT (no cambiaba nada) y además el setter nativo fallaba bajo trimming AOT
(`ArgumentNullException: ptr`). Se revirtió; `GraphOptimizationLevel.ORT_ENABLE_ALL` permanece.

**Caveat del entorno:** con poca RAM libre, Windows llegó a matar el proceso durante la carga de 1,6 GB
(stdout cerrado sin excepción). Mitigación operativa: carga perezosa + instancia residente del servicio
OpenCode + no lanzar cargas consecutivas en benchmarks.

## 8. Servidor MCP (src/Laya.Mcp)

- **Transporte stdio** (`"type": "local"`): OpenCode gestiona el ciclo de vida (spawn/reciclaje); cero
  servicios, puertos ni procesos huérfanos. Upgrade documentado: el mismo host compila con
  `ModelContextProtocol.AspNetCore` para Streamable HTTP si aparece concurrencia multi-cliente real.
- **SDK oficial `ModelContextProtocol` 2.2.0**: `AddMcpServer().WithStdioServerTransport().WithTools<T>()`.
  Notas API reales: la clase de tools debe ser de **instancia** (inyección por constructor primario);
  **no existe `McpProtocolException`** en v2 → los errores de validación se capturan y devuelven como
  JSON `{"error": …}` (el LLM cliente recibe el detalle y puede reintentar); el logging va a stderr
  para no contaminar el canal.
- **Resolución de artefactos:** `--model-dir` → `LAYA_MODEL_DIR` → `./artifacts` junto al exe →
  `%USERPROFILE%\.cache\laya\artifacts`. `LayaModel.FindModel` acepta `model.onnx`, `model_fp32.onnx`,
  `model_int8.onnx` (en ese orden), lo que hace que apuntar a `artifacts\yehor` sirva para la
  validación cruzada sin configuración extra.
- **JSON source-generated** (`LayaJsonContext`) y opciones manuales: cero warnings de trim en publish.

## 9. Integración en OpenCode

- Config global: `mcp.servers.laya` local apuntando al exe AOT con `--model-dir …\artifacts\ti3x`.
- Agentes `gh-automation` / `gh-investigate` / `ms-docs` documentan el formato por tipo y el gating:
  actuar solo con `actProbability ≥ 0.6`; `confidence < 0.5` o `warnings` → provisional.
- Skill `laya-tools` y comando `/laya-triage` como materialización de los flujos de triage.
- **Plugin `laya-gate` (fase shadow):** hook oficial `permission.evaluate` (verificado en
  `@opencode/plugin@2.0.16`: `Permission.Effect = allow|deny|ask`) que clasifica comandos shell con
  Laya (`noul` "destructive?") y **solo anota** en el mensaje del permiso. Fail-closed por diseño:
  error/timeout → `ask` intacto; nunca `deny` automático.
  - **Medición que motiva la fase shadow:** la tarea "destructivo" está fuera de los 4 flujos del
    checkpoint. P(destructive) medida: seguros 0,34-0,42 vs destructivos 0,51-0,58 — hueco limpio
    pero escala aplastada por `noul:2` (T=1,98) y `act`=1,0 en todos. Umbral empírico 0,45, validación
    con datos reales antes de cualquier auto-aprobación.

## 10. Verificación y fixtures

Los tests son la fuente de verdad ejecutable (`tests/Laya.Tests`, 9/9):

| Test | Qué prueba |
|---|---|
| `Tokenizer_Reproduces_Reference_InputIds` | input_ids/attention/marcadores/qtype bit a bit vs `reference.npz` |
| `Config_Temperatures_Match_LayaConfig` | buckets exactos/rango/default 1.0 |
| `Inference_Matches_Reference_Logits_And_ActProbs` | logits ±1e-4 y act_probs ±1e-5 (3 ejecuciones batch-1) |
| `EndToEnd_Decode_Matches_Reference_Json` × {ti3x, fp32, int8} | decode completo vs golden, tolerancias por modelo |
| `Yehor_Logits_Match_Reference` × {fp32, int8} | validación cruzada del segundo exportador |
| `Tokenizer_Known_Ids_And_RoundTrip` | IDs especiales + round-trip NFC |

Lector `.npy/.npz` propio (`tests/Laya.Tests/Infra/Npy.cs`) — sin numpy/Python en el pipeline.

## 11. Trabajo futuro (F4)

- Recalibración de temperaturas propia (issue #186) — entonces `confidence` podría ser umbral firme.
- Fine-tune/export multilingüe si el uso en español lo justifica.
- Salida de `laya-gate` de shadow a `autoApprove` solo tras evidencia acumulada en su log de storage.
- Evaluar Streamable HTTP (`ModelContextProtocol.AspNetCore`) si aparecen clientes MCP concurrentes.
