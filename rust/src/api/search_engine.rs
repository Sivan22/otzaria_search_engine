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
    schema: Schema,
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
        let segment = schema_builder.add_u64_field("segment", STORED);
        let isPdf = schema_builder.add_bool_field("isPdf", STORED);
        let file_path = schema_builder.add_text_field("filePath", TEXT | STORED);
        let schema = schema_builder.build();
        let index = Index::open_or_create(path, schema.clone());
        let index = index.expect("Failed to create index").clone();
        let index_reader = index.reader().expect("Failed to create index reader");
        let index_writer = index
            .writer(50_000_000)
            .expect("Failed to create index writer");

        SearchEngine {
            path: path.to_string(),
            index: index,
            schema: schema,
            index_writer: index_writer,
            index_reader: index_reader,
        }
    }

    pub fn add_document(
        &mut self,
        _id: u64,
        _title: &str,
        _text: &str,
        _segment: u64,
        _isPdf: bool,
        _filePath: &str,
    ) -> Result<()> {
        let title = self.schema.get_field("title").unwrap();
        let text = self.schema.get_field("text").unwrap();
        let id = self.schema.get_field("id").unwrap();
        let segment = self.schema.get_field("segment").unwrap();
        let isPdf = self.schema.get_field("isPdf").unwrap();
        let file_path = self.schema.get_field("filePath").unwrap();

        self.index_writer.add_document(doc!(
        title => _title,
        text => _text,
        id => _id,
        segment => _segment,
        isPdf => _isPdf,
        file_path => _filePath
        ))?;
        Ok(())
    }
    pub fn search(&mut self, query: &str, books: &Vec<String>) -> String {
        String::from("")
    }
}
