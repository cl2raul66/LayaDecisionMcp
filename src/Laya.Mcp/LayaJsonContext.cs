using System.Text.Json.Serialization;

namespace Laya.Mcp;

/// <summary>Contexto System.Text.Json source-generated (AOT-safe) para los resultados de las tools.</summary>
[JsonSerializable(typeof(LayaToolDecision))]
[JsonSerializable(typeof(LayaToolError))]
[JsonSerializable(typeof(LayaTemperaturesResult))]
[JsonSerializable(typeof(LayaInfoResult))]
internal sealed partial class LayaJsonContext : JsonSerializerContext;