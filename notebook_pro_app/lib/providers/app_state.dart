import 'dart:async';
import 'dart:io' show Process, ProcessSignal, Platform, Directory, File;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:window_manager/window_manager.dart';
import '../services/api_service.dart';
import '../services/studio_service.dart';

// import '../services/firebase_service.dart'; // Uncomment when ready to use Firebase

class AppState extends ChangeNotifier with WindowListener {
  final ApiService _apiService = ApiService();
  // final FirebaseService _firebaseService = FirebaseService(); // Uncomment when ready

  // Current state
  List<Space> _spaces = [];
  Space? _currentSpace;
  List<ChatInfo> _chats = [];
  Chat? _currentChat;
  List<Map<String, dynamic>> _uploadedFiles = [];
  String _activeTool = 'Chat';
  bool _isLoading = false;
  String? _error;
  String? _successMessage;
  bool _isUploading = false;
  bool _isThinking = false;

  // UI state
  bool _sidebarOpen = true;
  bool _plusMenuOpen = false;
  bool _toolsMenuOpen = false;
  bool _filesDrawerOpen = false;

  // Getters
  List<Space> get spaces => _spaces;
  Space? get currentSpace => _currentSpace;
  List<ChatInfo> get chats => _chats;
  Chat? get currentChat => _currentChat;
  List<Map<String, dynamic>> get uploadedFiles => _uploadedFiles;
  String get activeTool => _activeTool;
  bool get isLoading => _isLoading;
  String? get error => _error;
  String? get successMessage => _successMessage;
  bool get isUploading => _isUploading;
  bool get isThinking => _isThinking;
  bool get sidebarOpen => _sidebarOpen;
  bool get plusMenuOpen => _plusMenuOpen;
  bool get toolsMenuOpen => _toolsMenuOpen;
  bool get filesDrawerOpen => _filesDrawerOpen;

  // Backend process tracking
  Process? _backendProcess;
  StreamSubscription<List<int>>? _backendStdoutSub;
  StreamSubscription<List<int>>? _backendStderrSub;
  bool _backendAutoStarted = false;
  bool _backendStopping = false;
  static const List<int> _backendPorts = [8011, 8010, 8001, 8000];
  Future<bool>? _backendStartupFuture;
  Timer? _uploadHealthWatchdog;
  Timer? _windowStateDebounce;
  int _uploadHealthMisses = 0;
  bool _uploadHealthProbeInFlight = false;
  bool _windowListenerRegistered = false;

  static const String _prefWindowWidth = 'desktop_window_width';
  static const String _prefWindowHeight = 'desktop_window_height';
  static const String _prefWindowX = 'desktop_window_x';
  static const String _prefWindowY = 'desktop_window_y';
  static const String _prefWindowMaximized = 'desktop_window_maximized';

  Future<void> _startUploadHealthWatchdog() async {
    _stopUploadHealthWatchdog();

    final url = await ApiService.getBaseUrl();
    _uploadHealthWatchdog = Timer.periodic(const Duration(seconds: 15), (
      timer,
    ) async {
      if (!_isUploading) {
        timer.cancel();
        return;
      }

      if (_uploadHealthProbeInFlight) {
        return;
      }

      _uploadHealthProbeInFlight = true;
      try {
        final healthy = await ApiService.testConnection(url);
        if (healthy) {
          _uploadHealthMisses = 0;
          return;
        }

        _uploadHealthMisses += 1;
        // 4 missed probes (~60s) is treated as delayed heartbeat, not immediate failure.
        if (_uploadHealthMisses == 4) {
          _successMessage =
              '⚠️ Upload is still running, but backend heartbeat is delayed. Keep the app open.';
          notifyListeners();
        }
      } finally {
        _uploadHealthProbeInFlight = false;
      }
    });
  }

  void _stopUploadHealthWatchdog() {
    _uploadHealthWatchdog?.cancel();
    _uploadHealthWatchdog = null;
    _uploadHealthMisses = 0;
    _uploadHealthProbeInFlight = false;
  }

  bool get _isDesktopPlatform {
    if (kIsWeb) {
      return false;
    }
    return Platform.isWindows || Platform.isMacOS || Platform.isLinux;
  }

  Future<void> _restoreWindowState() async {
    if (!_isDesktopPlatform) {
      return;
    }

    if (!_windowListenerRegistered) {
      windowManager.addListener(this);
      _windowListenerRegistered = true;
    }

    try {
      final prefs = await SharedPreferences.getInstance();
      final width = prefs.getDouble(_prefWindowWidth);
      final height = prefs.getDouble(_prefWindowHeight);
      final x = prefs.getDouble(_prefWindowX);
      final y = prefs.getDouble(_prefWindowY);
      final wasMaximized = prefs.getBool(_prefWindowMaximized) ?? false;

      if (width != null && height != null) {
        await windowManager.setSize(Size(width, height));
      }

      if (x != null && y != null) {
        await windowManager.setPosition(Offset(x, y));
      }

      if (wasMaximized) {
        await windowManager.maximize();
      }
    } catch (_) {
      // Best-effort restore only.
    }
  }

  Future<void> _persistWindowState() async {
    if (!_isDesktopPlatform) {
      return;
    }

    try {
      final prefs = await SharedPreferences.getInstance();
      final maximized = await windowManager.isMaximized();
      await prefs.setBool(_prefWindowMaximized, maximized);

      if (!maximized) {
        final size = await windowManager.getSize();
        final position = await windowManager.getPosition();
        await prefs.setDouble(_prefWindowWidth, size.width);
        await prefs.setDouble(_prefWindowHeight, size.height);
        await prefs.setDouble(_prefWindowX, position.dx);
        await prefs.setDouble(_prefWindowY, position.dy);
      }
    } catch (_) {
      // Best-effort persist only.
    }
  }

  void _scheduleWindowStatePersist() {
    if (!_isDesktopPlatform) {
      return;
    }

    _windowStateDebounce?.cancel();
    _windowStateDebounce = Timer(const Duration(milliseconds: 600), () {
      unawaited(_persistWindowState());
    });
  }

  @override
  void onWindowClose() async {
    await _persistWindowState();
    await _stopManagedBackendProcess();
    await windowManager.destroy();
  }

  @override
  void onWindowMove() {
    _scheduleWindowStatePersist();
  }

  @override
  void onWindowResize() {
    _scheduleWindowStatePersist();
  }

  @override
  void onWindowMaximize() {
    _scheduleWindowStatePersist();
  }

  @override
  void onWindowUnmaximize() {
    _scheduleWindowStatePersist();
  }

  Future<void> _killProcessTree(Process process) async {
    try {
      process.kill(ProcessSignal.sigterm);
    } catch (_) {
      // Best effort.
    }

    try {
      await process.exitCode.timeout(const Duration(seconds: 2));
      return;
    } catch (_) {
      // Escalate below.
    }

    if (Platform.isWindows) {
      try {
        await Process.run('taskkill', [
          '/F',
          '/T',
          '/PID',
          process.pid.toString(),
        ]);
      } catch (_) {
        // Best-effort cleanup only.
      }
      return;
    }

    try {
      process.kill(ProcessSignal.sigkill);
    } catch (_) {
      // Best effort.
    }
  }

  Future<void> _stopManagedBackendProcess() async {
    if (_backendStopping) {
      return;
    }

    final process = _backendProcess;
    if (!_backendAutoStarted || process == null) {
      return;
    }

    _backendStopping = true;
    try {
      final stdoutSub = _backendStdoutSub;
      final stderrSub = _backendStderrSub;
      _backendStdoutSub = null;
      _backendStderrSub = null;

      if (stdoutSub != null) {
        await stdoutSub.cancel();
      }
      if (stderrSub != null) {
        await stderrSub.cancel();
      }

      await _killProcessTree(process);
    } finally {
      _backendProcess = null;
      _backendAutoStarted = false;
      _backendStopping = false;
    }
  }

  // ==================== Backend Management ====================

  List<String> _candidateBackendUrls(String currentUrl) {
    final candidates = <String>[];

    void addCandidate(String value) {
      if (value.isNotEmpty && !candidates.contains(value)) {
        candidates.add(value);
      }
    }

    final normalized = currentUrl.toLowerCase();
    final currentIsLocal =
        normalized.contains('localhost') || normalized.contains('127.0.0.1');

    // For local desktop runs, probe known local ports first so the app can
    // discover a manually started backend_nvidia instance on 8011.
    if (!currentIsLocal) {
      addCandidate(currentUrl);
    }

    for (final port in _backendPorts) {
      addCandidate('http://127.0.0.1:$port');
      addCandidate('http://localhost:$port');
    }

    if (currentIsLocal) {
      addCandidate(currentUrl);
    }

    return candidates;
  }

  Future<void> _switchBackendUrl(String url, {bool showBanner = true}) async {
    final currentUrl = await ApiService.getBaseUrl();
    if (currentUrl == url) {
      return;
    }

    await ApiService.setBaseUrl(url);
    if (!showBanner) {
      return;
    }

    _successMessage = 'Backend found locally. Switched to $url';
    notifyListeners();

    Future.delayed(const Duration(seconds: 3), () {
      if (_successMessage != null &&
          _successMessage!.contains('Backend found locally')) {
        _successMessage = null;
        notifyListeners();
      }
    });
  }

  Future<bool> _tryStartBackendOnPort({
    required String executablePath,
    required List<String> executableArgs,
    required String backendPath,
    required int port,
  }) async {
    final processEnvironment = Map<String, String>.from(Platform.environment)
      ..['PYTHONIOENCODING'] = 'utf-8'
      ..['PYTHONUTF8'] = '1'
      ..['PYTHONNOUSERSITE'] = '1';

    final process = await Process.start(
      executablePath,
      [...executableArgs, '--host', '0.0.0.0', '--port', '$port'],
      workingDirectory: backendPath,
      runInShell: false,
      environment: processEnvironment,
    );

    final stdoutSub = process.stdout.listen((data) {
      print('Backend[$port]: ${String.fromCharCodes(data)}');
    });

    final stderrSub = process.stderr.listen((data) {
      print('Backend diagnostic[$port]: ${String.fromCharCodes(data)}');
    });

    final startupUrls = ['http://127.0.0.1:$port', 'http://localhost:$port'];
    var exited = false;
    process.exitCode.then((_) {
      exited = true;
    });

    // Model preload can take longer than a normal web server startup.
    const startupDeadline = Duration(seconds: 60);
    final start = DateTime.now();

    while (!exited && DateTime.now().difference(start) < startupDeadline) {
      for (final url in startupUrls) {
        try {
          if (await ApiService.testConnection(url)) {
            await _switchBackendUrl(url, showBanner: false);
            final oldStdout = _backendStdoutSub;
            final oldStderr = _backendStderrSub;
            if (oldStdout != null) {
              await oldStdout.cancel();
            }
            if (oldStderr != null) {
              await oldStderr.cancel();
            }
            _backendStdoutSub = stdoutSub;
            _backendStderrSub = stderrSub;
            _backendProcess = process;
            _backendAutoStarted = true;
            return true;
          }
        } catch (_) {
          // Keep checking remaining URLs.
        }
      }

      await Future.delayed(const Duration(milliseconds: 500));
    }

    await stdoutSub.cancel();
    await stderrSub.cancel();
    await _killProcessTree(process);
    return false;
  }

  Future<bool> _checkBackendRunning() async {
    final currentUrl = await ApiService.getBaseUrl();

    for (final url in _candidateBackendUrls(currentUrl)) {
      try {
        final reachable = await ApiService.testConnection(url);
        if (reachable) {
          await _switchBackendUrl(url);
          return true;
        }
      } catch (_) {
        // Keep checking candidates.
      }
    }

    return false;
  }

  Future<bool> _startBackendIfNeeded() async {
    if (_backendStartupFuture != null) {
      return _backendStartupFuture!;
    }

    _backendStartupFuture = _startBackendIfNeededInternal();
    try {
      return await _backendStartupFuture!;
    } finally {
      _backendStartupFuture = null;
    }
  }

  Future<bool> _startBackendIfNeededInternal() async {
    // Check if backend is already running
    final isRunning = await _checkBackendRunning();
    if (isRunning) {
      print('✅ Backend is already running');
      return true;
    }

    // On web, mobile, or non-desktop platforms, we can't auto-start backend
    // But don't show an error yet - let the app try to connect first
    if (kIsWeb) {
      print('⚠️ Running on web - cannot auto-start backend');
      return false; // Backend not running yet, but don't block initialization
    }

    if (!Platform.isWindows && !Platform.isMacOS && !Platform.isLinux) {
      print('📱 Running on mobile - cannot auto-start backend');
      print('   User should configure backend URL in Settings');
      return false; // On mobile, user needs to configure URL in settings
    }

    // Try to auto-start backend on desktop
    try {
      print('🚀 Starting backend server...');

      // Get project root (assuming app is in notebook_pro_app folder)
      final currentDir = Directory.current.path;
      String projectRoot;

      if (currentDir.endsWith('notebook_pro_app')) {
        projectRoot = currentDir.substring(
          0,
          currentDir.lastIndexOf('notebook_pro_app'),
        );
      } else if (currentDir.contains('notebook_pro_app')) {
        projectRoot = currentDir.substring(
          0,
          currentDir.indexOf('notebook_pro_app'),
        );
      } else {
        // Try parent directory
        projectRoot = Directory.current.parent.path;
      }

      String executablePath;
      List<String> executableArgs;
      String backendPath;

      if (Platform.isWindows) {
        final installedRoot = File(Platform.resolvedExecutable).parent.path;
        final packagedBackend =
            '$installedRoot\\backend\\notebookpro_backend.exe';
        final embeddedPython = '$installedRoot\\python\\python.exe';

        if (File(packagedBackend).existsSync()) {
          executablePath = packagedBackend;
          executableArgs = const [];
          backendPath = '$installedRoot\\backend';
        } else if (File(embeddedPython).existsSync()) {
          executablePath = embeddedPython;
          executableArgs = const ['-m', 'uvicorn', 'main:app'];
          backendPath = '$installedRoot\\backend';
        } else {
          executablePath = '$projectRoot\virt\Scripts\python.exe';
          executableArgs = const ['-m', 'uvicorn', 'main:app'];
          final preferred = '$projectRoot\backend_nvidia';
          final fallback = '$projectRoot\backend';
          backendPath = Directory(preferred).existsSync()
              ? preferred
              : fallback;
        }
      } else {
        executablePath = '$projectRoot/virt/bin/python';
        executableArgs = const ['-m', 'uvicorn', 'main:app'];
        final preferred = '$projectRoot/backend_nvidia';
        final fallback = '$projectRoot/backend';
        backendPath = Directory(preferred).existsSync() ? preferred : fallback;
      }

      // Check if paths exist
      if (!File(executablePath).existsSync()) {
        throw Exception('Backend executable not found at: $executablePath');
      }
      if (!Directory(backendPath).existsSync()) {
        throw Exception('Backend not found at: $backendPath');
      }

      print('Using backend executable: $executablePath');
      print('Backend path: $backendPath');

      for (final port in _backendPorts) {
        print('Trying backend startup on port $port...');
        final started = await _tryStartBackendOnPort(
          executablePath: executablePath,
          executableArgs: executableArgs,
          backendPath: backendPath,
          port: port,
        );

        if (started) {
          final activeUrl = await ApiService.getBaseUrl();
          print('✅ Backend server started successfully on $activeUrl');
          _successMessage = '✅ Backend server started automatically';
          notifyListeners();

          // Auto-dismiss success message
          Future.delayed(const Duration(seconds: 3), () {
            _successMessage = null;
            notifyListeners();
          });
          return true;
        }
      }

      // Recovery path: backend might already be running on one of the candidates.
      final alreadyRunning = await _checkBackendRunning();
      if (alreadyRunning) {
        print('✅ Backend already running; reusing existing process');
        return true;
      }

      throw Exception('Backend started but not responding');
    } catch (e) {
      // Recovery path: if auto-start failed due to port collision, reuse existing backend if reachable.
      final alreadyRunning = await _checkBackendRunning();
      if (alreadyRunning) {
        print('✅ Backend already running; skipping auto-start');
        return true;
      }

      print('❌ Failed to start backend: $e');
      _setError(
        'Could not start backend automatically.\n\nPlease start it manually:\n1. Open a terminal\n2. cd backend\n3. ..\\virt\\Scripts\\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000\n\nOr use: start.bat',
      );
      return false;
    }
  }

  // ==================== Initialization ====================

  Future<void> initialize() async {
    _setLoading(true);
    try {
      await _restoreWindowState();

      // Try to start backend if needed (or check if it's running)
      final backendAvailable = await _startBackendIfNeeded();

      // Try to load spaces regardless - backend might come online or user might start it
      try {
        await loadSpaces();
        if (_spaces.isNotEmpty) {
          await selectSpace(_spaces.first.id);
        }
      } catch (e) {
        // Show platform-appropriate error message
        if (!backendAvailable) {
          // Check if we're on mobile/web
          if (kIsWeb ||
              (!Platform.isWindows && !Platform.isMacOS && !Platform.isLinux)) {
            // Mobile or web - show settings hint
            _setError(
              'Cannot connect to backend server.\n\nTap the Settings icon (⚙️) in the top-right to configure your backend URL.',
            );
          } else {
            // Desktop - show manual start instructions
            _setError(
              '⚠️ Backend not running!\n\nPlease start the backend server manually:\n\n1. Open a terminal\n2. Run: cd backend\n3. Run: ..\\virt\\Scripts\\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000\n\nOr use the startup script: start.bat',
            );
          }
        } else {
          _setError('Failed to load data: ${e.toString()}');
        }
      }
    } catch (e) {
      _setError(e.toString());
    } finally {
      _setLoading(false);
    }
  }

  @override
  void dispose() {
    _stopUploadHealthWatchdog();
    _windowStateDebounce?.cancel();
    unawaited(_persistWindowState());
    if (_windowListenerRegistered) {
      windowManager.removeListener(this);
      _windowListenerRegistered = false;
    }
    unawaited(_stopManagedBackendProcess());
    super.dispose();
  }

  // ==================== Spaces ====================

  Future<void> loadSpaces() async {
    try {
      _spaces = await _apiService.getSpaces();
      notifyListeners();
    } catch (e) {
      // Provide helpful error message for connection issues
      String errorMsg = e.toString().toLowerCase();
      if (errorMsg.contains('clientexception') ||
          errorMsg.contains('socketexception') ||
          errorMsg.contains('connection') ||
          errorMsg.contains('failed host lookup') ||
          errorMsg.contains('network')) {
        // Connection error - provide platform-specific guidance
        if (kIsWeb ||
            (!Platform.isWindows && !Platform.isMacOS && !Platform.isLinux)) {
          // Mobile/Web - show settings hint
          throw Exception(
            'Connection failed. Please configure your backend URL in Settings (\u2699\ufe0f).',
          );
        } else {
          // Desktop - might be localhost issue
          throw Exception(
            'Cannot connect to backend. Please ensure it is running.',
          );
        }
      } else {
        // Re-throw original error for other types of failures
        rethrow;
      }
    }
  }

  Future<void> createSpace(String name) async {
    _setLoading(true);
    try {
      final space = await _apiService.createSpace(name);
      _spaces.add(space);
      await selectSpace(space.id);
    } catch (e) {
      _setError('Failed to create space: $e');
    } finally {
      _setLoading(false);
    }
  }

  Future<void> selectSpace(String spaceId) async {
    _currentSpace = _spaces.firstWhere((s) => s.id == spaceId);
    _currentChat = null;
    await loadChats();
    await loadUploadedFiles();
    notifyListeners();
  }

  Future<void> deleteSpace(String spaceId) async {
    try {
      await _apiService.deleteSpace(spaceId);
      _spaces.removeWhere((s) => s.id == spaceId);
      if (_currentSpace?.id == spaceId) {
        _currentSpace = _spaces.isNotEmpty ? _spaces.first : null;
        if (_currentSpace != null) {
          await selectSpace(_currentSpace!.id);
        }
      }
      notifyListeners();
    } catch (e) {
      _setError('Failed to delete space: $e');
    }
  }

  // ==================== Chats ====================

  Future<void> loadChats() async {
    if (_currentSpace == null) return;

    try {
      _chats = await _apiService.getChats(_currentSpace!.id);
      notifyListeners();
    } catch (e) {
      _setError('Failed to load chats: $e');
    }
  }

  Future<void> loadChat(String chatId) async {
    if (_currentSpace == null) return;

    try {
      _currentChat = await _apiService.getChat(_currentSpace!.id, chatId);
      notifyListeners();
    } catch (e) {
      _setError('Failed to load chat: $e');
    }
  }

  Future<void> newChat() async {
    _currentChat = null;
    notifyListeners();
  }

  Future<void> deleteChat(String chatId) async {
    if (_currentSpace == null) return;

    try {
      await _apiService.deleteChat(_currentSpace!.id, chatId);
      _chats.removeWhere((c) => c.id == chatId);
      if (_currentChat?.id == chatId) {
        _currentChat = null;
      }
      notifyListeners();
    } catch (e) {
      _setError('Failed to delete chat: $e');
    }
  }

  Future<void> deleteAssistantResponse({required int messageIndex}) async {
    if (_currentSpace == null || _currentChat == null) return;
    if (messageIndex < 0 || messageIndex >= _currentChat!.messages.length)
      return;

    final message = _currentChat!.messages[messageIndex];
    int relatedUserIndex = -1;
    for (int i = messageIndex - 1; i >= 0; i--) {
      if (_currentChat!.messages[i].role == 'user') {
        relatedUserIndex = i;
        break;
      }
    }

    if (message.role != 'assistant') {
      _setError('Only assistant responses can be deleted.');
      return;
    }

    try {
      if (_currentChat!.id.isNotEmpty) {
        await _apiService.deleteChatMessage(
          spaceId: _currentSpace!.id,
          chatId: _currentChat!.id,
          timestamp: message.timestamp,
          role: message.role,
          content: message.content,
          deleteRelatedUser: true,
        );
      }

      final updatedMessages = List<ChatMessage>.from(_currentChat!.messages);
      final indexesToRemove = <int>{messageIndex};
      if (relatedUserIndex >= 0) {
        indexesToRemove.add(relatedUserIndex);
      }

      final sortedIndexes = indexesToRemove.toList()
        ..sort((a, b) => b.compareTo(a));
      for (final idx in sortedIndexes) {
        if (idx >= 0 && idx < updatedMessages.length) {
          updatedMessages.removeAt(idx);
        }
      }

      _currentChat = Chat(
        id: _currentChat!.id,
        messages: updatedMessages,
        createdAt: _currentChat!.createdAt,
        updatedAt: DateTime.now().toIso8601String(),
      );

      await loadChats();
      _showTemporarySuccess(
        relatedUserIndex >= 0
            ? 'Question and response deleted'
            : 'Response deleted',
      );
      notifyListeners();
    } catch (e) {
      _setError('Failed to delete response: $e');
    }
  }

  // ==================== Messaging ====================

  Future<void> sendMessage(String query) async {
    if (_currentSpace == null) return;

    _setLoading(true);
    _isThinking = true;
    notifyListeners();

    try {
      // Add user message to UI immediately
      if (_currentChat == null) {
        _currentChat = Chat(
          id: '',
          messages: [],
          createdAt: DateTime.now().toIso8601String(),
          updatedAt: DateTime.now().toIso8601String(),
        );
      }

      _currentChat!.messages.add(
        ChatMessage(
          role: 'user',
          content: query,
          timestamp: DateTime.now().toIso8601String(),
          sources: null,
        ),
      );

      // Add an empty assistant placeholder so streamed tokens can render immediately.
      final placeholderTimestamp = DateTime.now().toIso8601String();
      _currentChat!.messages.add(
        ChatMessage(
          role: 'assistant',
          content: '',
          timestamp: placeholderTimestamp,
          sources: const [],
          suggestedFollowUps: const [],
        ),
      );
      notifyListeners();

      String streamedText = '';
      String finalChatId = _currentChat!.id;
      String finalTimestamp = placeholderTimestamp;
      List<dynamic> finalSources = const [];
      List<String> finalFollowUps = const [];
      String? finalNotice;

      void updateAssistantMessage({
        required String content,
        String? timestamp,
        List<dynamic>? sources,
        List<String>? followUps,
      }) {
        if (_currentChat == null || _currentChat!.messages.isEmpty) return;

        final updatedMessages = List<ChatMessage>.from(_currentChat!.messages);
        final idx = updatedMessages.length - 1;
        final prev = updatedMessages[idx];

        if (prev.role != 'assistant') return;

        updatedMessages[idx] = ChatMessage(
          role: 'assistant',
          content: content,
          timestamp: timestamp ?? prev.timestamp,
          sources: sources ?? prev.sources,
          suggestedFollowUps: followUps ?? prev.suggestedFollowUps,
        );

        _currentChat = Chat(
          id: finalChatId,
          messages: updatedMessages,
          createdAt: _currentChat!.createdAt,
          updatedAt: timestamp ?? DateTime.now().toIso8601String(),
        );
      }

      final stream = _apiService.sendMessageStream(
        query: query,
        spaceId: _currentSpace!.id,
        chatId: _currentChat?.id.isEmpty == true ? null : _currentChat?.id,
        workflow: _activeTool.toLowerCase(),
      );

      await for (final event in stream) {
        if (event.event == 'meta') {
          finalChatId = (event.data['chat_id'] as String?) ?? finalChatId;
          finalTimestamp =
              (event.data['timestamp'] as String?) ?? finalTimestamp;
          finalSources = event.data['sources'] is List
              ? List<dynamic>.from(event.data['sources'])
              : finalSources;
          finalNotice = event.data['ui_notice'] as String?;
          continue;
        }

        if (event.event == 'token') {
          final delta = (event.data['delta'] as String?) ?? '';
          if (delta.isNotEmpty) {
            streamedText += delta;
            updateAssistantMessage(
              content: streamedText,
              timestamp: finalTimestamp,
              sources: finalSources,
              followUps: finalFollowUps,
            );
            notifyListeners();
          }
          continue;
        }

        if (event.event == 'followups') {
          finalFollowUps = event.data['suggested_follow_ups'] is List
              ? List<String>.from(event.data['suggested_follow_ups'])
              : finalFollowUps;
          continue;
        }

        if (event.event == 'done') {
          finalChatId = (event.data['chat_id'] as String?) ?? finalChatId;
          finalTimestamp =
              (event.data['timestamp'] as String?) ?? finalTimestamp;
          final fallbackResponse = (event.data['response'] as String?) ?? '';
          finalSources = event.data['sources'] is List
              ? List<dynamic>.from(event.data['sources'])
              : finalSources;
          finalNotice = (event.data['ui_notice'] as String?) ?? finalNotice;
          finalFollowUps = event.data['suggested_follow_ups'] is List
              ? List<String>.from(event.data['suggested_follow_ups'])
              : finalFollowUps;

          if (streamedText.trim().isEmpty &&
              fallbackResponse.trim().isNotEmpty) {
            streamedText = fallbackResponse;
          }

          updateAssistantMessage(
            content: streamedText,
            timestamp: finalTimestamp,
            sources: finalSources,
            followUps: finalFollowUps,
          );
          notifyListeners();
          continue;
        }

        if (event.event == 'error') {
          final message =
              (event.data['message'] as String?) ?? 'Unknown streaming error';
          throw Exception(message);
        }
      }

      final isQuotaError = streamedText.startsWith(
        'Error generating response: Gemini quota exceeded',
      );
      if (isQuotaError) {
        _error =
            'Gemini usage limit reached. Try again shortly or use backup provider settings.';
        _isThinking = false;
        notifyListeners();
        return;
      }

      if (_currentChat != null) {
        _currentChat = Chat(
          id: finalChatId,
          messages: List<ChatMessage>.from(_currentChat!.messages),
          createdAt: _currentChat!.createdAt,
          updatedAt: finalTimestamp,
        );
      }

      // Show backend notices (quota fallback, model switch, etc.) as toast/banner.
      if (finalNotice != null && finalNotice.isNotEmpty) {
        _successMessage = finalNotice;
        notifyListeners();

        Future.delayed(const Duration(seconds: 4), () {
          if (_successMessage == finalNotice) {
            _successMessage = null;
            notifyListeners();
          }
        });
      }

      // Reload chats list
      await loadChats();
      _isThinking = false;
      notifyListeners();
    } catch (e) {
      if (_currentChat != null && _currentChat!.messages.isNotEmpty) {
        final updated = List<ChatMessage>.from(_currentChat!.messages);
        final last = updated.last;
        if (last.role == 'assistant' && last.content.trim().isEmpty) {
          updated.removeLast();
          _currentChat = Chat(
            id: _currentChat!.id,
            messages: updated,
            createdAt: _currentChat!.createdAt,
            updatedAt: DateTime.now().toIso8601String(),
          );
        }
      }
      _isThinking = false;
      _setError('Failed to send message: $e');
    } finally {
      _setLoading(false);
    }
  }

  Future<void> addAssistantReplyToNotebook({
    required String question,
    required String answer,
    String? assistantTimestamp,
    List<String>? tags,
  }) async {
    if (_currentSpace == null) {
      _setError('Please select a space before adding notes to notebook.');
      return;
    }

    try {
      await StudioService.addChatToNotebook(
        spaceId: _currentSpace!.id,
        spaceName: _currentSpace!.name,
        question: question,
        answer: answer,
        chatId: _currentChat?.id,
        assistantTimestamp: assistantTimestamp,
        tags: tags ?? const ['chat'],
      );

      _successMessage = '✅ Added to ${_currentSpace!.name} notebook';
      notifyListeners();

      Future.delayed(const Duration(seconds: 3), () {
        if (_successMessage != null && _successMessage!.contains('Added to')) {
          _successMessage = null;
          notifyListeners();
        }
      });
    } catch (e) {
      _setError('Failed to add to notebook: $e');
    }
  }

  Future<void> downloadAssistantReplyAsPdf({
    required String question,
    required String answer,
  }) async {
    if (_currentSpace == null) {
      _setError('Please select a space before exporting PDF.');
      return;
    }

    try {
      final result = await StudioService.exportSingleAnswerPdf(
        spaceId: _currentSpace!.id,
        question: question,
        answer: answer,
        title: 'answer_${DateTime.now().millisecondsSinceEpoch}',
      );

      final savePath = await FilePicker.saveFile(
        dialogTitle: 'Save answer PDF',
        fileName: result.filename,
        type: FileType.custom,
        allowedExtensions: const ['pdf'],
      );

      if (savePath == null || savePath.isEmpty) {
        return;
      }

      await File(savePath).writeAsBytes(result.bytes, flush: true);
      _showTemporarySuccess('PDF saved successfully');
    } catch (e) {
      _setError('Failed to export PDF: $e');
    }
  }

  Future<void> downloadCurrentChatAnswersAsPdf() async {
    if (_currentSpace == null) {
      _setError('Please select a space before exporting PDF.');
      return;
    }
    if (_currentChat == null || _currentChat!.messages.isEmpty) {
      _setError('No chat answers available to export.');
      return;
    }

    final items = <Map<String, String>>[];
    String lastQuestion = '';

    for (final message in _currentChat!.messages) {
      if (message.role == 'user') {
        lastQuestion = message.content;
      } else if (message.role == 'assistant') {
        items.add({
          'question': lastQuestion.isNotEmpty ? lastQuestion : 'Question',
          'answer': message.content,
        });
      }
    }

    if (items.isEmpty) {
      _setError('No assistant answers available to export in this chat.');
      return;
    }

    try {
      final result = await StudioService.exportCombinedAnswersPdf(
        spaceId: _currentSpace!.id,
        chatTitle: _currentChat!.id.isEmpty
            ? 'combined_answers_${DateTime.now().millisecondsSinceEpoch}'
            : 'chat_${_currentChat!.id}_answers',
        items: items,
      );

      final savePath = await FilePicker.saveFile(
        dialogTitle: 'Save combined answers PDF',
        fileName: result.filename,
        type: FileType.custom,
        allowedExtensions: const ['pdf'],
      );

      if (savePath == null || savePath.isEmpty) {
        return;
      }

      await File(savePath).writeAsBytes(result.bytes, flush: true);
      _showTemporarySuccess('Combined PDF saved successfully');
    } catch (e) {
      _setError('Failed to export combined PDF: $e');
    }
  }

  void _showTemporarySuccess(String message) {
    _successMessage = message;
    notifyListeners();

    Future.delayed(const Duration(seconds: 3), () {
      if (_successMessage == message) {
        _successMessage = null;
        notifyListeners();
      }
    });
  }

  // ==================== File Upload ====================

  Future<void> loadUploadedFiles() async {
    if (_currentSpace == null) return;

    try {
      _uploadedFiles = await _apiService.getUploadedFiles(_currentSpace!.id);
      notifyListeners();
    } catch (e) {
      // Silently fail - files list is non-critical
      _uploadedFiles = [];
    }
  }

  Future<void> deleteFile(String filename) async {
    if (_currentSpace == null) return;

    try {
      await _apiService.deleteFile(_currentSpace!.id, filename);
      _uploadedFiles.removeWhere((f) => f['filename'] == filename);
      await loadSpaces(); // Refresh file count
      _successMessage = 'âœ… File "$filename" deleted successfully';
      notifyListeners();

      // Auto-dismiss success message after 3 seconds
      Future.delayed(const Duration(seconds: 3), () {
        _successMessage = null;
        notifyListeners();
      });
    } catch (e) {
      _setError('Failed to delete file: $e');
    }
  }

  Future<void> uploadFiles(List<PlatformFile> files) async {
    if (_currentSpace == null) return;

    _isUploading = true;
    _successMessage = null;
    _error = null;
    notifyListeners();

    try {
      await _startUploadHealthWatchdog();
      final result = await _apiService.uploadFiles(_currentSpace!.id, files);
      _successMessage =
          '✅ Successfully uploaded ${result['files_processed']} files (${result['total_chunks']} chunks processed)';
      await loadSpaces(); // Refresh file count
      await loadUploadedFiles(); // Refresh files list
      notifyListeners();

      // Auto-dismiss success message after 5 seconds
      Future.delayed(const Duration(seconds: 5), () {
        if (_successMessage != null) {
          _successMessage = null;
          notifyListeners();
        }
      });
    } catch (e) {
      _setError('❌ Failed to upload files: $e');
    } finally {
      _stopUploadHealthWatchdog();
      _isUploading = false;
      notifyListeners();
    }
  }

  // ==================== Tools ====================

  void setActiveTool(String tool) {
    _activeTool = tool;
    _toolsMenuOpen = false;
    notifyListeners();
  }

  // ==================== UI State ====================

  void toggleSidebar() {
    _sidebarOpen = !_sidebarOpen;
    notifyListeners();
  }

  void setSidebarOpen(bool open) {
    _sidebarOpen = open;
    notifyListeners();
  }

  void togglePlusMenu() {
    _plusMenuOpen = !_plusMenuOpen;
    _toolsMenuOpen = false;
    notifyListeners();
  }

  void toggleToolsMenu() {
    _toolsMenuOpen = !_toolsMenuOpen;
    _plusMenuOpen = false;
    notifyListeners();
  }

  void toggleFilesDrawer() {
    _filesDrawerOpen = !_filesDrawerOpen;
    notifyListeners();
  }

  void closeMenus() {
    _plusMenuOpen = false;
    _toolsMenuOpen = false;
    notifyListeners();
  }

  // ==================== Helpers ====================

  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void _setError(String? error) {
    _error = error;
    notifyListeners();
  }

  void showError(String error) {
    _error = error;
    notifyListeners();
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }

  void showSuccess(String message) {
    _successMessage = message;
    notifyListeners();
  }

  void clearSuccessMessage() {
    _successMessage = null;
    notifyListeners();
  }
}
