// This file is automatically generated, so please do not edit it.
// Generated by `flutter_rust_bridge`@ 2.3.0.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import '../frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

// These function are ignored because they are on traits that is not defined in current crate (put an empty `#[frb]` on it to unignore): `clone`

String testBindings({required String name}) =>
    RustLib.instance.api.crateApiSearchEngineTestBindings(name: name);

// Rust type: RustOpaqueMoi<flutter_rust_bridge::for_generated::RustAutoOpaqueInner<Box < dyn Query >>>
abstract class BoxQuery implements RustOpaqueInterface {}

// Rust type: RustOpaqueMoi<flutter_rust_bridge::for_generated::RustAutoOpaqueInner<Index>>
abstract class Index implements RustOpaqueInterface {}

// Rust type: RustOpaqueMoi<flutter_rust_bridge::for_generated::RustAutoOpaqueInner<SearchEngine>>
abstract class SearchEngine implements RustOpaqueInterface {
  Future<void> addDocument(
      {required BigInt id,
      required String title,
      required String text,
      required BigInt segment,
      required bool isPdf,
      required String filePath});

  Future<void> commit();

  static Future<BoxQuery> createSearchQuery(
          {required Index index,
          required String searchTerm,
          required List<String> bookTitles,
          required bool fuzzy}) =>
      RustLib.instance.api.crateApiSearchEngineSearchEngineCreateSearchQuery(
          index: index,
          searchTerm: searchTerm,
          bookTitles: bookTitles,
          fuzzy: fuzzy);

  // HINT: Make it `#[frb(sync)]` to let it become the default constructor of Dart class.
  static Future<SearchEngine> newInstance({required String path}) =>
      RustLib.instance.api.crateApiSearchEngineSearchEngineNew(path: path);

  Future<List<SearchResult>> search(
      {required String query,
      required List<String> books,
      required int limit,
      required bool fuzzy});

  Stream<List<SearchResult>> searchStream(
      {required String query,
      required List<String> books,
      required int limit,
      required bool fuzzy});
}

class SearchResult {
  final String title;
  final String text;
  final BigInt id;
  final BigInt segment;
  final bool isPdf;
  final String filePath;

  const SearchResult({
    required this.title,
    required this.text,
    required this.id,
    required this.segment,
    required this.isPdf,
    required this.filePath,
  });

  @override
  int get hashCode =>
      title.hashCode ^
      text.hashCode ^
      id.hashCode ^
      segment.hashCode ^
      isPdf.hashCode ^
      filePath.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SearchResult &&
          runtimeType == other.runtimeType &&
          title == other.title &&
          text == other.text &&
          id == other.id &&
          segment == other.segment &&
          isPdf == other.isPdf &&
          filePath == other.filePath;
}
