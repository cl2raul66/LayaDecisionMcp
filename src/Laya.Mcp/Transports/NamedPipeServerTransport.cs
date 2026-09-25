using System.IO.Pipes;
using System.Text.Json;
using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using ModelContextProtocol;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Laya.Mcp.Transports;

/// <summary>
/// Named Pipe transport for MCP server - allows multiple clients to connect to a single server instance.
/// Windows: NamedPipeServerStream (\\.\pipe\LayaMcp)
/// Linux/macOS: Would use UnixDomainSocket (not implemented here, Windows-only for now).
/// </summary>
public sealed class NamedPipeServerTransport : ITransport, IAsyncDisposable
{
    private readonly string _pipeName;
    private readonly ILogger _logger;
    private readonly Channel<JsonRpcMessage> _messageChannel = Channel.CreateUnbounded<JsonRpcMessage>();
    private readonly CancellationTokenSource _cts = new();
    private readonly List<NamedPipeClientSession> _sessions = [];
    private readonly object _sessionsLock = new();
    private Task? _listenerTask;
    private volatile bool _disposed;

    public string? SessionId => null; // Multi-session transport, no single session ID
    public ChannelReader<JsonRpcMessage> MessageReader => _messageChannel.Reader;

    public NamedPipeServerTransport(string pipeName, ILoggerFactory? loggerFactory = null)
    {
        _pipeName = pipeName ?? throw new ArgumentNullException(nameof(pipeName));
        _logger = loggerFactory?.CreateLogger<NamedPipeServerTransport>() ?? NullLogger<NamedPipeServerTransport>.Instance;
    }

    /// <summary>
    /// Starts listening for client connections on the named pipe.
    /// </summary>
    public void StartListening()
    {
        if (_listenerTask is not null)
            throw new InvalidOperationException("Transport is already listening.");

        _listenerTask = Task.Run(ListenAsync, _cts.Token);
        _logger.LogInformation("Named Pipe server listening on {PipeName}", _pipeName);
    }

    private async Task ListenAsync()
    {
        while (!_cts.Token.IsCancellationRequested)
        {
            NamedPipeServerStream? pipeServer = null;
            try
            {
                pipeServer = new NamedPipeServerStream(
                    _pipeName,
                    PipeDirection.InOut,
                    NamedPipeServerStream.MaxAllowedServerInstances,
                    PipeTransmissionMode.Message,
                    PipeOptions.Asynchronous);

                _logger.LogDebug("Waiting for client connection on {PipeName}...", _pipeName);
                await pipeServer.WaitForConnectionAsync(_cts.Token);

                if (_cts.Token.IsCancellationRequested)
                    break;

                _logger.LogInformation("Client connected to {PipeName}", _pipeName);

                var session = new NamedPipeClientSession(pipeServer, _logger, _cts.Token);
                lock (_sessionsLock)
                {
                    _sessions.Add(session);
                }

                // Start handling this client's messages
                _ = Task.Run(() => HandleClientAsync(session), _cts.Token);
            }
            catch (OperationCanceledException) when (_cts.Token.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex) when (!_cts.Token.IsCancellationRequested)
            {
                _logger.LogError(ex, "Error accepting client connection on {PipeName}", _pipeName);
                pipeServer?.Dispose();
                await Task.Delay(100, _cts.Token); // Brief delay before retry
            }
        }
    }

    private async Task HandleClientAsync(NamedPipeClientSession session)
    {
        try
        {
            await session.ReadMessagesAsync(_messageChannel.Writer, _cts.Token);
        }
        catch (OperationCanceledException) when (_cts.Token.IsCancellationRequested)
        {
            // Expected on shutdown
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling client session");
        }
        finally
        {
            lock (_sessionsLock)
            {
                _sessions.Remove(session);
            }
            await session.DisposeAsync();
            _logger.LogInformation("Client disconnected from {PipeName}", _pipeName);
        }
    }

    public async Task SendMessageAsync(JsonRpcMessage message, CancellationToken cancellationToken = default)
    {
        if (_disposed)
            throw new InvalidOperationException("Transport has been disposed.");

        // For multi-session transport, we need to know which session to send to.
        // The message context should contain the session information.
        // For now, broadcast to all connected sessions (simplified approach).
        // A more sophisticated implementation would route based on session ID.
        
        List<NamedPipeClientSession> sessionsToSend;
        lock (_sessionsLock)
        {
            sessionsToSend = _sessions.ToList();
        }

        var tasks = sessionsToSend.Select(s => s.SendMessageAsync(message, cancellationToken).AsTask());
        await Task.WhenAll(tasks);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        _cts.Cancel();

        if (_listenerTask is not null)
        {
            try { await _listenerTask; } catch { /* ignore */ }
        }

        List<NamedPipeClientSession> sessionsToDispose;
        lock (_sessionsLock)
        {
            sessionsToDispose = _sessions.ToList();
            _sessions.Clear();
        }

        await Task.WhenAll(sessionsToDispose.Select(s => s.DisposeAsync().AsTask()));
        _messageChannel.Writer.Complete();
        
        _logger.LogInformation("Named Pipe server stopped on {PipeName}", _pipeName);
    }
}

/// <summary>
/// Represents a single client session over a named pipe connection.
/// </summary>
internal sealed class NamedPipeClientSession : IAsyncDisposable
{
    private readonly NamedPipeServerStream _pipe;
    private readonly ILogger _logger;
    private readonly CancellationToken _shutdownToken;
    private readonly StreamReader _reader;
    private readonly StreamWriter _writer;
    private volatile bool _disposed;

    public NamedPipeClientSession(NamedPipeServerStream pipe, ILogger logger, CancellationToken shutdownToken)
    {
        _pipe = pipe;
        _logger = logger;
        _shutdownToken = shutdownToken;
        _reader = new StreamReader(pipe, leaveOpen: true);
        _writer = new StreamWriter(pipe, leaveOpen: true) { AutoFlush = true };
    }

    public async Task ReadMessagesAsync(ChannelWriter<JsonRpcMessage> messageChannel, CancellationToken cancellationToken)
    {
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _shutdownToken);

        while (!linkedCts.Token.IsCancellationRequested)
        {
            string? line;
            try
            {
                line = await _reader.ReadLineAsync();
            }
            catch (IOException) when (linkedCts.Token.IsCancellationRequested)
            {
                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }

            if (line == null) // EOF
                break;

            if (string.IsNullOrWhiteSpace(line))
                continue;

            try
            {
                var message = JsonSerializer.Deserialize<JsonRpcMessage>(line, McpJsonUtilities.DefaultOptions);
                if (message is not null)
                {
                    await messageChannel.WriteAsync(message, linkedCts.Token);
                }
            }
            catch (JsonException ex)
            {
                _logger.LogWarning(ex, "Failed to deserialize JSON-RPC message: {Line}", line);
            }
        }
    }

    public async ValueTask SendMessageAsync(JsonRpcMessage message, CancellationToken cancellationToken)
    {
        if (_disposed)
            return;

        try
        {
            var json = JsonSerializer.Serialize(message, McpJsonUtilities.DefaultOptions);
            await _writer.WriteLineAsync(json);
            await _writer.FlushAsync(cancellationToken);
        }
        catch (Exception ex) when (!_disposed)
        {
            _logger.LogError(ex, "Failed to send message to client");
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        try
        {
            _writer.Dispose();
            _reader.Dispose();
            _pipe.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error disposing client session");
        }
    }
}

/// <summary>
/// Extension methods for registering the Named Pipe transport with the MCP server.
/// </summary>
public static class NamedPipeServerTransportExtensions
{
    /// <summary>
    /// Configures the MCP server to use Named Pipe transport instead of stdio.
    /// </summary>
    public static IMcpServerBuilder WithNamedPipeServerTransport(
        this IMcpServerBuilder builder,
        string pipeName = "LayaMcp")
    {
        builder.Services.AddSingleton<ITransport>(sp =>
        {
            var loggerFactory = sp.GetRequiredService<ILoggerFactory>();
            var transport = new NamedPipeServerTransport(pipeName, loggerFactory);
            transport.StartListening();
            return transport;
        });
        return builder;
    }
}