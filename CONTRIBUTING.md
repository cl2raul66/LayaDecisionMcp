# Contribuir a LayaDecisionMcp

Gracias por contribuir. Este documento define **cómo** se trabaja aquí; el **por qué** de cada
decisión técnica vive en [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md), que es lectura obligatoria antes de
tocar código.

> **Regla de oro**: `SYSTEM_DESIGN.md` es la fuente de verdad del comportamiento. Si tu cambio
> contradice una decisión documentada ahí (formato de secuencia, temperaturas, alias de salidas,
> tolerancias de los golden), primero discute la decisión en un issue y actualiza el documento en el
> mismo PR. Código y diseño nunca deben divergir.

## Requisitos

- .NET SDK **10.0.4xx** (`dotnet --version`).
- Visual Studio con herramientas C++ (necesarias para `PublishAot` en win-x64).
- PowerShell (Windows) o **pwsh 7** para `tools/*.ps1` (el script de inspección usa System.Text.Json de pwsh).
- **No se usa Python ni Node** en el flujo de build/test: los artefactos se descargan con PowerShell y
  los tests verifican contra fixtures ya exportados.

## Flujo de trabajo

```powershell
.\tools\fetch-artifacts.ps1        # descarga modelos + fixtures (~4 GB, una vez)
dotnet build LayaDecisionMcp.slnx  # la solución usa formato .slnx (.NET 10)
dotnet test tests\Laya.Tests       # 9 tests (5 golden + extras), verde obligatorio
dotnet publish src\Laya.Mcp\Laya.Mcp.csproj -c Release -r win-x64 -o publish\laya-mcp-win-x64
```

Antes de cada PR: compila con **0 warnings** (el trim/AOT no puede emitir ninguno) y `dotnet test`
en verde. Si cambias el servidor, repite el publish y el smoke stdio (`initialize → tools/list →
tools/call`); describe el resultado en el PR.

## Convenciones de código

- **C# idiomático moderno** (net10.0): constructores primarios, collection expressions, pattern
  matching. Código y comentarios en **español**; identificadores de dominio (nombres de tools,
  JSON, protocolo) en inglés porque forman parte del contrato MCP.
- **AOT/trim seguro, sin excepciones**:
  - Serialización JSON solo con `System.Text.Json` source-generated (`LayaJsonContext`); nunca
    `JsonSerializerOptions.Web` ni reflexión.
  - Regex solo con `[GeneratedRegex]` sobre clases `partial`.
  - Sin reflexión en el hot path de inferencia.
- **Tests dorados sobre fixtures**: cualquier cambio en tokenizer, plantilla o decode debe mantener
  `artifacts\ti3x\fixtures` (reference.npz/reference.json) como referencia exacta. Si añades soporte
  para otro exportador ONNX, añade su rama parametrizada con tolerancia documentada (ver
  `GoldenTests.cs`: fp32 comparte tolerancia con ti3x; int8 admite ±2e-2/±5e-2).
- **Errores de validación de tools**: devuélvelos como JSON `{"error": "…"}` con mensaje accionable
  para el LLM cliente (ver `LayaToolError`); no lances excepciones genéricas, que el SDK convierte en
  "An error occurred invoking …" sin detalle. Excepciones internas sí para fallos del servidor.
- **Carga perezosa del modelo**: el constructor del runtime debe seguir siendo barato; el modelo ONNX
  (~1,6 GB) solo se materializa en la primera decisión.

## Estructura del repo

```
src/Laya.Tokenizer/    ByteLevel BPE puro (tokenizer.json genérico, sin deps)
src/Laya.Inference/    LayaConfig / LayaSequence / LayaModel (ONNX Runtime, decode tipado)
src/Laya.Mcp/          Host MCP stdio (ModelContextProtocol SDK + Hosting), tools
tests/Laya.Tests/      xUnit golden contra fixtures (lector .npy/.npz propio)
tools/                 fetch-artifacts.ps1 (descarga) e inspect-laya.ps1 (análisis)
artifacts/             NO se commitean modelos ni tokenizers; sí fixtures y laya_config.json
publish/               Binarios AOT (gitignored)
```

## Commits y PRs

- Mensajes en español, imperativo, resumen concreto (`"Tokenizer: merges por rango mínimo"`).
- Un PR = una decisión. Si el cambio altera comportamiento observable (salidas, warnings, gating),
  incluye test y actualización de `SYSTEM_DESIGN.md`.
- Los benchmarks que afirmen rendimiento deben incluir tabla con condiciones (CPU, fp32/int8, frío/
  caliente) reproducibles con los scripts del propio repo.

## Qué NO cambiar sin abrir issue primero

- El formato de la secuencia de entrada (`[CLS] … [MASK] opción …`) incluido el **espacio inicial**
  antes de cada opción (el tokenizer ByteLevel lo codifica distinto; romperlo invalida los golden).
- Las temperaturas efectivas: se leen de `temperature_by_options`, no del array base.
- El contrato "act siempre normalizada" de `ExecuteRaw` (incluye el alias `act_logits` del exportador
  yehor con softmax aplicado dentro).
- El límite de 3 opciones (cabeza compartida `head_max_len` 256 del checkpoint).
