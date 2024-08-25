#[flutter_rust_bridge::frb(sync)] // Synchronous mode for simplicity of the demo
pub fn test_bindings(name: String) -> String {
    format!("Hello, {name}!")
}
use anyhow::{Error, Result};
use log::debug;
use serde_json::{json, Value};
use std::borrow::{Borrow, BorrowMut};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::{self, BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Score, Searcher};
use tantivy::{schema::*, Directory};

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
        let mmap_directory = MmapDirectory::open(path).expect("unable to open mmap directory");
        let index = Index::open_or_create(mmap_directory, schema.clone());
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
        self.index_writer.commit()?;
        Ok(())
    }
    pub fn search(&mut self, query: &str, books: &Vec<String>, limit: u32) -> Result<Vec<String>> {
        fn create_search_query(
            index: &Index,
            search_term: &str,
            book_titles: &[String],
        ) -> Result<Box<dyn tantivy::query::Query>> {
            let schema = index.schema();
            let text_field = schema.get_field("text").unwrap();
            let title_field = schema.get_field("title").unwrap();

            // Create the main text search query
            let text_query = TermQuery::new(
                Term::from_field_text(text_field, search_term),
                IndexRecordOption::WithFreqsAndPositions,
            );

            // Create a boolean query for the book titles
            let mut title_queries = Vec::new();
            for book_title in book_titles {
                let title_query = TermQuery::new(
                    Term::from_field_text(title_field, book_title),
                    IndexRecordOption::Basic,
                );
                title_queries.push((
                    Occur::Should,
                    Box::new(title_query) as Box<dyn tantivy::query::Query>,
                ));
            }
            let title_filter = BooleanQuery::new(title_queries);

            // Combine the text search and title filter
            let mut bool_query = BooleanQuery::new(vec![
                (
                    Occur::Must,
                    Box::new(text_query) as Box<dyn tantivy::query::Query>,
                ),
                (
                    Occur::Must,
                    Box::new(title_filter) as Box<dyn tantivy::query::Query>,
                ),
            ]);
            Ok(Box::new(bool_query))
        }

        let index = &self.index;
        let schema = &self.schema;
        let query = create_search_query(index, query, books).unwrap();
        let searcher = index.reader().unwrap().searcher();
        let top_docs = searcher
            .search(
                &query,
                &tantivy::collector::TopDocs::with_limit(limit as usize),
            )
            .unwrap();
        let mut results = Vec::<String>::new();

        for (_score, doc_address) in top_docs {
            // Retrieve the actual content of documents given its `doc_address`.
            let retrieved_doc = searcher
                .doc::<TantivyDocument>(doc_address)
                .expect("cannot find document");
            results.push(retrieved_doc.to_json(&schema));
        }

        Ok(results)
    }
}
