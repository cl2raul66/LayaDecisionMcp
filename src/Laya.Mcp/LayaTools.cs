using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using Laya.Inference;
using ModelContextProtocol.Server;

namespace Laya.Mcp;

/// <summary>Resultado tipado de una decisión, apto para que el agente actúe (con transparencia de calibración).</summary>
public sealed record LayaToolDecision(
    string Type,
    string Question,
    string? Choice,
    int? ChoiceIndex,
    double? Score,
    double? Noul,
    double[] Probabilities,
    double Confidence,
    double ActProbability,
    double Temperature,
    string TemperatureBucket,
    int SeqLen,
    string[] Warnings);

public sealed record LayaTemperaturesResult(
    int HeadMaxLen,
    int MaxLen,
    IReadOnlyDictionary<string, double> TemperatureByOptions,
    string DefaultWarning);

public sealed record LayaInfoResult(
    string ArtifactsDir,
    string ModelPath,
    long ModelBytes,
    int VocabSize,
    int HeadMaxLen,
    int MaxLen,
    string[] Caveats);

/// <summary>Error recuperable de validación (el LLM cliente puede corregirlo y reintentar).</summary>
public sealed record LayaToolError(string Error);

/// <summary>
/// Tools MCP del servidor Laya (decisiones tipadas choice/score/noul).
/// El modelo ONNX se carga perezosamente en la primera llamada.
/// </summary>
[McpServerToolType]
public sealed class LayaTools
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
    private static readonly LayaJsonContext Json = new(JsonOptions);

    private readonly LayaRuntime _runtime;

    public LayaTools(LayaRuntime runtime)
    {
        _runtime = runtime;
    }

    [McpServerTool]
    [Description(
        "Decisión tipada de Laya (modelo typed-decisions). " +
        "type: 'choice' (elige la mejor opción), 'score' (valor esperado sobre niveles 0..N-1) o 'noul' (probabilidad de que la afirmación se cumpla). " +
        "question: nombre corto del campo a decidir. instructions: instrucción natural de la tarea. " +
        "options: máx 3 opciones YA renderizadas — para choice/noul en formato 'etiqueta: descripción' (p. ej. \"billing: Payments and refunds\"); " +
        "para score en formato 'level N: etiqueta' (p. ej. \"level 1: medium\"). " +
        "state: contexto/estado (p. ej. JSON del ticket) opcional. " +
        "El modelo es especialista en inglés. Devuelve probabilidades post-temperatura, confidence (1−entropía normalizada), " +
        "score esperado o P(noul), y act_probability (probabilidad de actuar sin escalar a humano — actúa directamente si es alta, refuerza si es baja).")]
    public string Decision(
        [Description("choice | score | noul")] string type,
        [Description("Nombre corto del campo/pregunta (p. ej. 'department', 'urgency').")] string question,
        [Description("Instrucción natural de la tarea en inglés.")] string instructions,
        [Description("Opciones renderizadas (1-3). choice/noul: 'etiqueta: descripción'; score: 'level N: etiqueta'.")] string[] options,
        [Description("Contexto/estado (p. ej. JSON del ticket). Opcional.")] string? state = null)
    {
        try
        {
            return DecideCore(type, question, instructions, options, state);
        }
        catch (ArgumentException ex)
        {
            // Error recuperable: el mensaje llega al LLM como texto para que corrija y reintente.
            return JsonSerializer.Serialize(new LayaToolError(ex.Message), Json.LayaToolError);
        }
    }

    private string DecideCore(string type, string question, string instructions, string[] options, string? state)
    {
        LayaQType qtype;
        switch (type.Trim().ToLowerInvariant())
        {
            case "choice": qtype = LayaQType.Choice; break;
            case "score": qtype = LayaQType.Score; break;
            case "noul": qtype = LayaQType.Noul; break;
            default:
                // El servidor MCP convierte la excepción en error JSON-RPC de la tool, visible para el LLM.
                throw new ArgumentException($"type inválido '{type}'. Use 'choice', 'score' o 'noul'.", nameof(type));
        }

        if (options.Length is 0 or > LayaSequence.MaxOptions)
            throw new ArgumentException($"options debe tener entre 1 y {LayaSequence.MaxOptions} elementos (tiene {options.Length}).", nameof(options));
        if (string.IsNullOrWhiteSpace(question))
            throw new ArgumentException("question no puede estar vacío.", nameof(question));
        if (string.IsNullOrWhiteSpace(instructions))
            throw new ArgumentException("instructions no puede estar vacío.", nameof(instructions));

        var req = new LayaDecisionRequest(qtype, question, instructions, options, state);
        var d = _runtime.Model.Decide(req);

        var warnings = new List<string>();
        // Temperatura por defecto sin calibración oficial del checkpoint.
        if (d.Temperature == 1.0 && !_runtime.Config.TemperatureByOptions.ContainsKey(d.TemperatureBucket))
            warnings.Add($"Combinación '{type}:{options.Length}' sin temperatura calibrada en laya_config.json; se usó 1.0. " +
                         "Considere recalibrar (issue #186 de Laya) antes de fijar umbrales de confianza.");
        // Gating act/escalate: probabilidad baja de actuar → reforzar la decisión con un humano.
        if (d.ActProbability < 0.6)
            warnings.Add($"act_probability {d.ActProbability:F2} < 0.6: la cabeza del modelo sugiere escalar/verificar antes de actuar.");
        // Sin opción clara (empate o confianza baja) solo se informa; el agente decide el umbral.
        if (d.Confidence < 0.5)
            warnings.Add($"confidence {d.Confidence:F2} < 0.5: distribución poco concentrada; trate la respuesta con cautela.");

        var result = new LayaToolDecision(
            qtype.ToString().ToLowerInvariant(),
            d.Question,
            d.Choice,
            d.ChoiceIndex,
            d.Score,
            d.Noul,
            d.Probabilities,
            Math.Round(d.Confidence, 4),
            Math.Round(d.ActProbability, 4),
            d.Temperature,
            d.TemperatureBucket,
            d.SeqLen,
            warnings.ToArray());

        return JsonSerializer.Serialize(result, Json.LayaToolDecision);
    }

    [McpServerTool]
    [Description(
        "Temperaturas de calibración del checkpoint Laya (laya_config.json): bucket por tipo y nº de opciones. " +
        "Útil para calibrar umbrales de confianza o entender el escalado que se aplica a los logits antes del softmax.")]
    public string Temperatures()
    {
        var result = new LayaTemperaturesResult(
            _runtime.Config.HeadMaxLen,
            _runtime.Config.MaxLen,
            _runtime.Config.TemperatureByOptions,
            "Combinaciones sin bucket (p. ej. score:2, noul:3) usan temperatura 1.0 por defecto.");
        return JsonSerializer.Serialize(result, Json.LayaTemperaturesResult);
    }

    [McpServerTool]
    [Description(
        "Información del checkpoint Laya cargado (rutas de artefactos, tamaño del modelo, vocabulario, límites) " +
        "y limitaciones conocidas del modelo typed-decisions (inglés, contexto 1024, máx 3 opciones, recalibración pendiente).")]
    public string Info()
    {
        var modelBytes = File.Exists(_runtime.ModelPath) ? new FileInfo(_runtime.ModelPath).Length : 0;
        var result = new LayaInfoResult(
            _runtime.ArtifactsDir,
            _runtime.ModelPath,
            modelBytes,
            _runtime.Tokenizer.VocabSize,
            _runtime.Config.HeadMaxLen,
            _runtime.Config.MaxLen,
            [
                "Modelo especialista en inglés; los flujos en español deben traducir el prompt antes de decidir.",
                "Contexto máximo 1024 tokens; instrucciones + opciones + estado deben caber.",
                "Máximo 3 opciones por decisión (cabeza de marcadores compartida, head_max_len 256).",
                "Confianza sin recalibrar (issue #186 de Laya): use umbrales conservadores hasta recalibrar temperaturas.",
            ]);
        return JsonSerializer.Serialize(result, Json.LayaInfoResult);
    }
}