import 'package:flutter/material.dart';
import 'package:otzaria_search_engine/src/rust/api/search_engine.dart';
import 'package:otzaria_search_engine/src/rust/frb_generated.dart';

Future<void> main() async {
  await RustLib.init();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('flutter_rust_bridge quickstart')),
        body: Center(
          child: Text(
              'Action: Call Rust `greet("Tom")`\nResult: `${testBindings(name: "Tom")}`'),
        ),
      ),
    );
  }
}
