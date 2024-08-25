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
        appBar: AppBar(title: const Text('otzaria search engine playground')),
        body: Center(
          child: FutureBuilder<SearchEngine>(
              future: SearchEngine.newInstance(
                  path: "C:\\dev\\tantivy\\playground"),
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  return Text("success: ${snapshot.data!.runtimeType}");
                }
                if (snapshot.hasError) {
                  return Text('error: ${snapshot.error}');
                }
                return Text('loading index...');
              }),
        ),
      ),
    );
  }
}
