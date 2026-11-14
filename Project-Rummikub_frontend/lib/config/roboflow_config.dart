/// Roboflow Workflow "Rummikub v1 Logic" client settings.
///
/// Pass the secret at build/run time — never commit it:
/// `flutter run --dart-define=ROBOFLOW_API_KEY=your_key`
class RoboflowConfig {
  static const apiKey = String.fromEnvironment('ROBOFLOW_API_KEY');

  static const workspace = String.fromEnvironment(
    'ROBOFLOW_WORKSPACE',
    defaultValue: 'matan-kronfeld',
  );

  static const workflowId = String.fromEnvironment(
    'ROBOFLOW_WORKFLOW_ID',
    defaultValue: 'rummikub-v1-logic',
  );

  static const imageInputName = String.fromEnvironment(
    'ROBOFLOW_IMAGE_INPUT',
    defaultValue: 'image',
  );

  static const _confidenceCutoffRaw = String.fromEnvironment(
    'ROBOFLOW_CONFIDENCE_CUTOFF',
    defaultValue: '0.15',
  );

  static double get confidenceCutoff =>
      double.tryParse(_confidenceCutoffRaw) ?? 0.15;

  static const urlOverride = String.fromEnvironment('ROBOFLOW_WORKFLOW_URL');

  /// POST target from the workflow Deploy snippet.
  static String get runUrl {
    if (urlOverride.isNotEmpty) return urlOverride;
    return 'https://serverless.roboflow.com/$workspace/workflows/$workflowId';
  }

  /// Documented Serverless Workflow path; used if [runUrl] returns 404.
  static String get inferFallbackUrl {
    return 'https://serverless.roboflow.com/infer/workflows/$workspace/$workflowId';
  }

  static bool get hasApiKey => apiKey.isNotEmpty;

  /// `--dart-define=VISION_DEBUG=true` keeps the exact JPEGs and workflow JSON
  /// of the last recognition and shows a download button for them on the
  /// correction screen. Off by default: normal runs keep no debug data.
  static const debugDump = bool.fromEnvironment('VISION_DEBUG');
}
