using Laya.Mcp;
using Laya.Mcp.Transports;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

// Servidor MCP de Laya (decisiones tipadas choice/score/noul).
// Uso:
//   Laya.Mcp [--model-dir <ruta>]                    # stdio (default)
//   Laya.Mcp --pipe [--pipe-name <nombre>]           # Named Pipe
//   env LAYA_MODEL_DIR=<ruta> Laya.Mcp --pipe

var argsParsed = ParseArgs(args);

if (argsParsed.ShowHelp)
{
    ShowHelp();
    return;
}

var builder = Host.CreateApplicationBuilder(args);
// El logging va a stderr: stdout queda reservado al transporte MCP stdio.
builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);
builder.Services.AddSingleton(new LayaRuntime(argsParsed.ArtifactsDir));

var mcpBuilder = builder.Services.AddMcpServer().WithTools<LayaTools>();

if (argsParsed.UseNamedPipe)
{
    mcpBuilder.WithNamedPipeServerTransport(argsParsed.PipeName);
}
else
{
    mcpBuilder.WithStdioServerTransport();
}

await builder.Build().RunAsync();

static (string ArtifactsDir, bool UseNamedPipe, string PipeName, bool ShowHelp) ParseArgs(string[] args)
{
    string? modelDir = null;
    bool useNamedPipe = false;
    string pipeName = "LayaMcp";
    bool showHelp = false;

    for (var i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--help":
            case "-h":
                showHelp = true;
                break;
            case "--model-dir":
                if (i + 1 < args.Length)
                    modelDir = args[++i];
                break;
            case "--pipe":
                useNamedPipe = true;
                break;
            case "--pipe-name":
                if (i + 1 < args.Length)
                    pipeName = args[++i];
                break;
        }
    }

    string artifactsDir;
    if (showHelp)
    {
        artifactsDir = ""; // Won't be used
    }
    else if (!string.IsNullOrWhiteSpace(modelDir))
    {
        artifactsDir = modelDir;
    }
    else
    {
        var env = Environment.GetEnvironmentVariable("LAYA_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(env))
            artifactsDir = env;
        else
            artifactsDir = LayaRuntime.ResolveArtifactsDir([], AppContext.BaseDirectory);
    }

    return (artifactsDir, useNamedPipe, pipeName, showHelp);
}

static void ShowHelp()
{
    Console.WriteLine("""
        Laya.Mcp - Servidor MCP para decisiones tipadas (choice/score/noul)

        Uso:
          Laya.Mcp [OPCIONES]

        Opciones:
          --model-dir <ruta>     Directorio de artefactos (tokenizer.json, model.onnx, laya_config.json)
          --pipe                 Usar transporte Named Pipe en lugar de stdio
          --pipe-name <nombre>   Nombre del pipe (default: LayaMcp)
          --help, -h             Mostrar esta ayuda

        Variables de entorno:
          LAYA_MODEL_DIR         Directorio de artefactos (alternativa a --model-dir)

        Ejemplos:
          Laya.Mcp                                    # stdio, busca artefactos en ./artifacts
          Laya.Mcp --model-dir ./artifacts            # stdio, directorio explícito
          Laya.Mcp --pipe                             # Named Pipe (\\.\pipe\LayaMcp)
          Laya.Mcp --pipe --pipe-name MiPipe          # Named Pipe personalizado
        """);
}