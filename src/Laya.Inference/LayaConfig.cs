using System.Text.Json;

namespace Laya.Inference;

/// <summary>
/// Configuración del checkpoint Laya (laya_config.json). Las temperaturas efectivas que se aplican
/// en el decode son las de <c>temperature_by_options</c> (bucket por tipo y nº de opciones), verificadas
/// empíricamente contra reference.json/validation-fp32 del modelo ONNX. El array base "temperature"
/// se conserva solo como referencia.
/// </summary>
public sealed class LayaConfig
{
    private readonly Dictionary<string, double> _byOptions;

    public LayaConfig(int headMaxLen, int maxLen, IReadOnlyList<double> baseTemps, Dictionary<string, double> byOptions)
    {
        HeadMaxLen = headMaxLen;
        MaxLen = maxLen;
        Temperatures = baseTemps;
        _byOptions = byOptions;
    }

    public int HeadMaxLen { get; }
    public int MaxLen { get; }
    public IReadOnlyList<double> Temperatures { get; }

    public IReadOnlyDictionary<string, double> TemperatureByOptions => _byOptions;

    public static LayaConfig Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var r = doc.RootElement;

        var temps = new List<double>();
        foreach (var t in r.GetProperty("temperature").EnumerateArray()) temps.Add(t.GetDouble());

        var byOptions = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var p in r.GetProperty("temperature_by_options").EnumerateObject())
            byOptions[p.Name] = p.Value.GetDouble();

        return new LayaConfig(
            r.GetProperty("head_max_len").GetInt32(),
            r.GetProperty("max_len").GetInt32(),
            temps,
            byOptions);
    }

    /// <summary>
    /// Temperatura efectiva para un tipo ("choice"/"score"/"noul") y nº de opciones.
    /// Orden: coincidencia exacta ("choice:2"), después buckets de rango ("score:3-5", "choice:11+");
    /// si no hay entrada -> 1.0 (documentado como default razonable para combinaciones no calibradas).
    /// </summary>
    public (double Temp, string Bucket) GetTemperature(string type, int nOptions)
    {
        var key = $"{type}:{nOptions}";
        if (_byOptions.TryGetValue(key, out var exact)) return (exact, key);

        foreach (var (k, tv) in _byOptions)
        {
            var prefix = type + ":";
            if (!k.StartsWith(prefix, StringComparison.Ordinal)) continue;
            var range = k.Substring(prefix.Length);
            if (RangeContains(range, nOptions)) return (tv, k);
        }
        return (1.0, key);
    }

    private static bool RangeContains(string range, int n)
    {
        // "a-b" | "a+" | "n"
        var plus = range.IndexOf('+', StringComparison.Ordinal);
        if (plus >= 0) return n >= int.Parse(range[..plus]);
        var dash = range.IndexOf('-', StringComparison.Ordinal);
        if (dash >= 0)
        {
            var lo = int.Parse(range[..dash]);
            var hi = int.Parse(range[(dash + 1)..]);
            return n >= lo && n <= hi;
        }
        return n == int.Parse(range);
    }
}