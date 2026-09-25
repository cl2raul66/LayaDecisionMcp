# LayaDecisionMcp

Servidor MCP local que expone el modelo **Laya `typed-decisions`** como herramientas de
**decisión tipada** (`choice` / `score` / `noul`) para [OpenCode](https://opencode.ai) y cualquier
cliente MCP. Corre 100 % en tu máquina: sin API externa, sin coste por llamada, sin que tus datos
salgan del equipo.

> **¿Por qué existe?** Los agentes toman decisiones pequeñas constantemente (¿qué label poner a un
> issue?, ¿qué tan urgente es?, ¿esta afirmación se cumple?). Enviar cada una a un LLM conversacional
> cuesta latencia, tokens y fuga de contexto. Laya es un modelo *System-1* (ModernBERT, 421 M): clasifica
> en ~1 s sobre CPU con probabilidades por opción y una señal de *actuar vs. escalar*. Este servidor lo
> empaqueta como MCP estándar para enchufarlo a tus agentes.

## Tools expuestas

| Tool | Qué hace |
|---|---|
| `laya_decision(type, question, instructions, options, state?)` | Devuelve la decisión con probabilidades, `confidence`, `actProbability` (puerta actuar/escalar), temperatura aplicada y warnings de calibración |
| `laya_temperatures()` | Buckets de temperatura de calibración del checkpoint (`laya_config.json`) |
| `laya_info()` | Checkpoint cargado, límites y limitaciones conocidas |

**Tipos de decisión** — máximo **3 opciones** por llamada, inglés-only, contexto ≤ 1024 tokens:

- `choice`: elige 1 opción. Opciones como `"etiqueta: descripción"`.
- `score`: valor esperado sobre niveles `0..N-1`. Opciones como `"level N: etiqueta"`.
- `noul`: probabilidad de que una afirmación se cumpla. Opciones `"false: …"` / `"true: …"`.

Ejemplo real (fixture dorado del proyecto): un ticket de cargo duplicado con
`type=choice, question=department` devuelve `"billing: Payments and refunds"` con probabilidades
`[0.7876, 0.1270, 0.0854]` y `actProbability = 1.0` (seguro para actuar sin escalar).

## Quickstart

```powershell
# 1. Descargar artefactos del modelo (~4 GB, desde Hugging Face)
.\tools\fetch-artifacts.ps1

# 2. Verificar contra los fixtures dorados (9 tests)
dotnet test tests\Laya.Tests

# 3. Publicar el binario Native AOT (win-x64, ~14 MB + onnxruntime)
dotnet publish src\Laya.Mcp\Laya.Mcp.csproj -c Release -r win-x64 -o publish\laya-mcp-win-x64
```

Registro en OpenCode (`~/.config/opencode/opencode.jsonc`):

```jsonc
{
  "mcp": {
    "servers": {
      "laya": {
        "type": "local",
        "command": [
          "C:\\ruta\\a\\publish\\laya-mcp-win-x64\\Laya.Mcp.exe",
          "--model-dir", "C:\\ruta\\a\\artifacts\\ti3x"
        ]
      }
    }
  }
}
```

`opencode mcp list` → `✓ laya connected`.

## Rendimiento medido (CPU, batch-1, secuencia ~70 tokens)

| | Primera llamada (carga del modelo) | Decisión caliente |
|---|---|---|
| fp32 (por defecto) | ~13-32 s | **~0,4-0,9 s** |
| int8 (`ya disponible en artifacts\yehor`) | ~19 s | ~2,8 s (más lento en CPU, 3× menos RAM) |

## Limitaciones conocidas

- **Inglés only**: el checkpoint es especialista; traduce/resume el contexto antes de decidir.
- **`confidence` no está calibrada** (issue #186 del proyecto Laya): úsala como aviso, no como umbral
  de bloqueo. El gating fiable es `actProbability ≥ 0.6`.
- Combinaciones tipo×nº-opciones sin bucket calibrado usan temperatura 1.0 (se avisa en `warnings`).
- Máximo 3 opciones por llamada y 1024 tokens de contexto.

## Documentación

- [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) — arquitectura, decisiones técnicas justificadas, benchmarks y protocolo.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — cómo contribuir, convenciones y reglas del proyecto.

## Atribuciones

Los artefactos ONNX proceden de
[`ti3x-m/laya-typed-decisions-onnx`](https://huggingface.co/ti3x-m/laya-typed-decisions-onnx) (fuente
principal, con fixtures de referencia) y
[`yehor-oleksiuk/laya-typed-decisions-onnx`](https://huggingface.co/yehor-oleksiuk/laya-typed-decisions-onnx)
(validación cruzada fp32/int8), ambos exportados del checkpoint
[`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions).
`LayaDecisionMcp` solo implementa el runtime y el servidor; no redistribuye los pesos (se descargan
con `tools/fetch-artifacts.ps1`).

## Licencia

[MIT](LICENSE).
