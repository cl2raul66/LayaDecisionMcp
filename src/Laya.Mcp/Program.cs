using Laya.Mcp;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

// Servidor MCP stdio de Laya (decisiones tipadas choice/score/noul).
// Uso: Laya.Mcp [--model-dir <ruta>]  |  env LAYA_MODEL_DIR
var artifactsDir = LayaRuntime.ResolveArtifactsDir(args, AppContext.BaseDirectory);

var builder = Host.CreateApplicationBuilder(args);
// El logging va a stderr: stdout queda reservado al transporte MCP stdio.
builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);
builder.Services.AddSingleton(new LayaRuntime(artifactsDir));
builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithTools<LayaTools>();

await builder.Build().RunAsync();