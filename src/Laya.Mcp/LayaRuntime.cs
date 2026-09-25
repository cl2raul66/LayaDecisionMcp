using Laya.Inference;
using Laya.Tokenizer;

namespace Laya.Mcp;

/// <summary>
/// Runtime del checkpoint Laya: carga tokenizer (rápido) en arranque y el modelo ONNX
/// de forma perezosa (la carga ocupa ~ GB de RAM y segundos), más resolución de artefactos.
/// </summary>
public sealed class LayaRuntime
{
    public string ArtifactsDir { get; }
    public string ModelPath { get; }
    public LayaTokenizer Tokenizer { get; }
    public LayaConfig Config { get; }

    private readonly Lazy<LayaModel> _model;

    public LayaRuntime(string artifactsDir)
    {
        ArtifactsDir = Path.GetFullPath(artifactsDir);

        var tokPath = Path.Combine(ArtifactsDir, "tokenizer.json");
        var cfgPath = Path.Combine(ArtifactsDir, "laya_config.json");
        ModelPath = FindModel(ArtifactsDir);

        Tokenizer = new LayaTokenizer(tokPath);
        Config = LayaConfig.Load(cfgPath);
        _model = new Lazy<LayaModel>(() => new LayaModel(ModelPath, tokPath, cfgPath));
    }

    public LayaModel Model => _model.Value;

    private static string FindModel(string dir)
    {
        foreach (var name in new[] { "model.onnx", "model_fp32.onnx", "model_int8.onnx" })
        {
            var candidate = Path.Combine(dir, name);
            if (File.Exists(candidate)) return candidate;
        }
        throw new FileNotFoundException(
            "No se encontró un modelo ONNX (esperaba model.onnx / model_fp32.onnx / model_int8.onnx) en " + dir);
    }

    /// <summary>
    /// Orden de resolución del directorio de artefactos:
    /// 1) --model-dir &lt;ruta&gt;  2) env LAYA_MODEL_DIR  3) ./artifacts junto al exe  4) %USERPROFILE%\.cache\laya\artifacts
    /// </summary>
    public static string ResolveArtifactsDir(string[] args, string baseDir)
    {
        for (var i = 0; i + 1 < args.Length; i++)
            if (string.Equals(args[i], "--model-dir", StringComparison.OrdinalIgnoreCase))
                return args[i + 1];

        var env = Environment.GetEnvironmentVariable("LAYA_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(env)) return env;

        foreach (var candidate in new[]
                 {
                     Path.Combine(baseDir, "artifacts"),
                     Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "laya", "artifacts"),
                 })
        {
            if (Directory.Exists(candidate)) return candidate;
        }

        throw new InvalidOperationException(
            "No se encontró el directorio de artefactos Laya. Use --model-dir <ruta> o defina LAYA_MODEL_DIR.");
    }
}