#[flutter_rust_bridge::frb(sync)] // Synchronous mode for simplicity of the demo
pub fn test_bindings(name: String) -> String {
    format!("Hello, {name}!")
}
use anyhow::{Error, Result};
use futures::stream::{Stream, StreamExt};
use log::debug;
use serde_json::{json, Value};
use std::borrow::{Borrow, BorrowMut};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::{self, BooleanQuery, Occur, Query, QueryParser, TermQuery, TermSetQuery};
use tantivy::{doc, tokenizer, Index, IndexReader, IndexWriter, ReloadPolicy, Score, Searcher};
use tantivy::{schema::*, Directory};

pub struct SearchResult {
    pub title: String,
    pub text: String,
    pub id: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
}

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
        let text = schema_builder.add_text_field("text", TEXT | STORED | FAST);
        let title = schema_builder.add_text_field(
            "title",
            TextOptions::default()
                .set_indexing_options(
                    TextFieldIndexing::default()
                        .set_tokenizer("raw")
                        .set_fieldnorms(false),
                )
                .set_stored(),
        );
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
        _is_pdf: bool,
        _file_path: &str,
    ) -> Result<()> {
        let title = self.schema.get_field("title").unwrap();
        let text = self.schema.get_field("text").unwrap();
        let id = self.schema.get_field("id").unwrap();
        let segment = self.schema.get_field("segment").unwrap();
        let is_pdf = self.schema.get_field("isPdf").unwrap();
        let file_path = self.schema.get_field("filePath").unwrap();

        self.index_writer.add_document(doc!(
        title => _title,
        text => _text,
        id => _id,
        segment => _segment,
        is_pdf => _is_pdf,
        file_path => _file_path
        ))?;

        Ok(())
    }
    pub fn commit(&mut self) -> Result<()> {
        self.index_writer.commit()?;
        Ok(())
    }
    fn create_search_query(
        index: &Index,
        search_term: &str,
        book_titles: &[String],
    ) -> Result<Box<dyn tantivy::query::Query>> {
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let title_field = schema.get_field("title").unwrap();

        // Create the main text search query
        let text_query = QueryParser::for_index(&index, vec![text_field]);
        let text_query = text_query.parse_query(search_term).unwrap();

        // Create a TermSetQuery for exact matching of book titles
        let title_terms: Vec<Term> = book_titles
            .iter()
            .map(|title| Term::from_field_text(title_field, title))
            .collect();
        let title_filter = TermSetQuery::new(title_terms);

        // Combine the text search and title filter
        let bool_query = BooleanQuery::new(vec![
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

    pub fn search(
        &mut self,
        query: &str,
        books: &Vec<String>,
        limit: u32,
    ) -> Result<Vec<SearchResult>> {
        let index = &self.index;
        let schema = &self.schema;
        let query = Self::create_search_query(index, query, books).unwrap();
        let searcher = index.reader().unwrap().searcher();
        let top_docs = searcher
            .search(
                &query,
                &tantivy::collector::TopDocs::with_limit(limit as usize),
            )
            .unwrap();
        let mut results = Vec::<SearchResult>::new();
        let title_field = schema.get_field("title").unwrap();
        let text_field = schema.get_field("text").unwrap();
        let id_field = schema.get_field("id").unwrap();
        let segment_field = schema.get_field("segment").unwrap();
        let is_pdf_field = schema.get_field("isPdf").unwrap();
        let file_path_field = schema.get_field("filePath").unwrap();

        for (_score, doc_address) in top_docs {
            match searcher.doc::<TantivyDocument>(doc_address) {
                Ok(retrieved_doc) => {
                    let title = retrieved_doc
                        .get_first(title_field)
                        .and_then(|v| match v {
                            OwnedValue::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let text = retrieved_doc
                        .get_first(text_field)
                        .and_then(|v| match v {
                            OwnedValue::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let id = retrieved_doc
                        .get_first(id_field)
                        .and_then(|v| match v {
                            OwnedValue::U64(y) => Some(*y),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let segment = retrieved_doc
                        .get_first(segment_field)
                        .and_then(|v| match v {
                            OwnedValue::U64(y) => Some(*y),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let is_pdf = retrieved_doc
                        .get_first(is_pdf_field)
                        .and_then(|v| match v {
                            OwnedValue::Bool(y) => Some(*y),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let file_path = retrieved_doc
                        .get_first(file_path_field)
                        .and_then(|v| match v {
                            OwnedValue::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let result = SearchResult {
                        title,
                        text,
                        id,
                        segment,
                        is_pdf,
                        file_path,
                    };
                    results.push(result);
                }
                Err(_) => continue,
            }
        }
        Ok(results)
    }

    pub fn search_stream<'a>(
        &'a mut self,
        query: &'a str,
        books: &'a Vec<String>,
        limit: u32,
    ) -> Pin<Box<dyn Stream<Item = Result<SearchResult>> + 'a>> {
        Box::pin(async_stream::try_stream! {
            let index = &self.index;
            let schema = &self.schema;
            let query = Self::create_search_query(index, query, books)?;
            let searcher = index.reader()?.searcher();
            let top_docs = searcher.search(
                &query,
                &tantivy::collector::TopDocs::with_limit(limit as usize),
            )?;

            let title_field = schema.get_field("title").unwrap();
            let text_field = schema.get_field("text").unwrap();
            let id_field = schema.get_field("id").unwrap();
            let segment_field = schema.get_field("segment").unwrap();
            let is_pdf_field = schema.get_field("isPdf").unwrap();
            let file_path_field = schema.get_field("filePath").unwrap();

            for (_score, doc_address) in top_docs {
                if let Ok(retrieved_doc) = searcher.doc::<TantivyDocument>(doc_address) {
                    let title = retrieved_doc
                        .get_first(title_field)
                        .and_then(|v| match v {
                            OwnedValue::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let text = retrieved_doc
                        .get_first(text_field)
                        .and_then(|v| match v {
                            OwnedValue::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let id = retrieved_doc
                        .get_first(id_field)
                        .and_then(|v| match v {
                            OwnedValue::U64(y) => Some(*y),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let segment = retrieved_doc
                        .get_first(segment_field)
                        .and_then(|v| match v {
                            OwnedValue::U64(y) => Some(*y),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let is_pdf = retrieved_doc
                        .get_first(is_pdf_field)
                        .and_then(|v| match v {
                            OwnedValue::Bool(y) => Some(*y),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let file_path = retrieved_doc
                        .get_first(file_path_field)
                        .and_then(|v| match v {
                            OwnedValue::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let result = SearchResult {
                        title,
                        text,
                        id,
                        segment,
                        is_pdf,
                        file_path,
                    };
                    yield result;
                }
            }
        })
    }
}
