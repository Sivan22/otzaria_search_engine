#[flutter_rust_bridge::frb(sync)] // Synchronous mode for simplicity of the demo
pub fn test_bindings(name: String) -> String {
    format!("Hello, {name}!")
}
use anyhow::{Error, Result};
use log::debug;
use std::borrow::{Borrow, BorrowMut};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::query::{Query, QueryParser, TermQuery};
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Score, Searcher};

pub struct SearchEngine {
    path: String,
    index: Index,
    index_writer: IndexWriter,
    index_reader: IndexReader,
}

impl SearchEngine {
    pub fn new(path: &str) -> Self {
        debug!("new path={}", path,);
        let schema_builder = Schema::builder();
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT | STORED);
        let title = schema_builder.add_text_field("title", TEXT);
        let id = schema_builder.add_u64_field("id", STORED);
        let line = schema_builder.add_u64_field("line", STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_dir(path, schema.clone());
        let index = index.expect("Failed to create index").clone();
        let index_reader = index.reader().expect("Failed to create index reader");
        let mut index_writer = index
            .writer(50_000_000)
            .expect("Failed to create index writer");

        SearchEngine {
            path: path.to_string(),
            index: index,
            index_writer: index_writer,
            index_reader: index_reader,
        }
    }

    pub fn index_text(&mut self, id: u64, title: &str, text: &str, line: u64) -> Result<()> {
        Ok(())
    }
    pub fn search(&mut self, query: &str, books: &Vec<String>) -> String {
        String::from("")
    }
}
