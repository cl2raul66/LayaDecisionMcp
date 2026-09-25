namespace Laya.Tests.Infra;

/// <summary>Localiza el directorio de artefactos (repo/artifacts) o usa LAYA_ARTIFACTS_DIR.</summary>
public static class ArtifactPaths
{
    public static string Root { get; } = ResolveRoot();

    public static string Ti3x => Path.Combine(Root, "ti3x");
    public static string Yehor => Path.Combine(Root, "yehor");

    public static string TokenizerJson => Path.Combine(Ti3x, "tokenizer.json");
    public static string LayaConfigJson => Path.Combine(Ti3x, "laya_config.json");
    public static string ModelOnnx => Path.Combine(Ti3x, "model.onnx");
    public static string ModelOnnxData => Path.Combine(Ti3x, "model.onnx_data");
    public static string YehorFp32 => Path.Combine(Yehor, "model_fp32.onnx");
    public static string YehorInt8 => Path.Combine(Yehor, "model_int8.onnx");
    public static string ReferenceNpz => Path.Combine(Ti3x, "fixtures", "reference.npz");
    public static string ReferenceJson => Path.Combine(Ti3x, "fixtures", "reference.json");

    private static string ResolveRoot()
    {
        var env = Environment.GetEnvironmentVariable("LAYA_ARTIFACTS_DIR");
        if (!string.IsNullOrWhiteSpace(env) && Directory.Exists(env)) return env;

        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "artifacts");
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("No se encontró el directorio 'artifacts' (defina LAYA_ARTIFACTS_DIR).");
    }
}