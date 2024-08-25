// This file is automatically generated, so please do not edit it.
// Generated by `flutter_rust_bridge`@ 2.3.0.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import '../frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

String testBindings({required String name}) =>
    RustLib.instance.api.crateApiSearchEngineTestBindings(name: name);

// Rust type: RustOpaqueMoi<flutter_rust_bridge::for_generated::RustAutoOpaqueInner<SearchEngine>>
abstract class SearchEngine implements RustOpaqueInterface {
  Future<void> indexText(
      {required BigInt id,
      required String title,
      required String text,
      required BigInt line});

  // HINT: Make it `#[frb(sync)]` to let it become the default constructor of Dart class.
  static Future<SearchEngine> newInstance({required String path}) =>
      RustLib.instance.api.crateApiSearchEngineSearchEngineNew(path: path);

  Future<String> search({required String query, required List<String> books});
}
