/// Coerces Firestore / JSON maps to `Map<String, dynamic>`.
Map<String, dynamic> asStringKeyedMap(dynamic value) {
  if (value is Map<String, dynamic>) return value;
  if (value is Map) {
    return value.map((key, nested) => MapEntry(key.toString(), nested));
  }
  throw ArgumentError('Expected a JSON object, got ${value.runtimeType}');
}

List<Map<String, dynamic>> asMapList(dynamic value) {
  if (value is! List) {
    throw ArgumentError('Expected a JSON array, got ${value.runtimeType}');
  }
  return value.map(asStringKeyedMap).toList();
}
