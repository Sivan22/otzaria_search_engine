#[flutter_rust_bridge::frb(sync)] // Synchronous mode for simplicity of the demo
pub fn test_bindings(name: String) -> String {
    format!("Hello, {name}!")
}
use crate::frb_generated::StreamSink;
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
use tantivy::index::Index;
use tantivy::query::{self, BooleanQuery, Occur, QueryParser, TermQuery, TermSetQuery};
use tantivy::query::{PhraseQuery, Query};
use tantivy::{
    doc, tokenizer, DocAddress, IndexReader, IndexWriter, Order, ReloadPolicy, Score, Searcher,
    SnippetGenerator,
};
use tantivy::{schema::*, Directory};

#[derive(Clone)]
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
        let id = schema_builder.add_u64_field("id", STORED | FAST);
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
    pub fn create_search_query(
        index: &Index,
        search_term: &str,
        book_titles: &[String],
        fuzzy: bool,
    ) -> Result<Box<dyn Query>> {
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let title_field = schema.get_field("title").unwrap();

        // Create the main text search query
        let mut text_query: Box<dyn Query> = {
            // in case of fuzzy search, use a query parser with fuzzy query
            if fuzzy {
                let mut text_query = QueryParser::for_index(&index, vec![text_field]);
                text_query.set_conjunction_by_default();
                text_query.set_field_fuzzy(text_field, false, 1, true);
                let text_query = text_query.parse_query(search_term).unwrap();
                Box::new(text_query) as Box<dyn Query>
            // in case of exact search, use a term query
            } else {
                Box::new(
                    QueryParser::for_index(&index, vec![text_field])
                        .parse_query(search_term)
                        .unwrap(),
                ) as Box<dyn Query>
            }
        };

        // Create a TermSetQuery for exact matching of book titles
        let title_terms: Vec<Term> = book_titles
            .iter()
            .map(|title| Term::from_field_text(title_field, title))
            .collect();
        let title_filter = TermSetQuery::new(title_terms);

        // Combine the text search and title filter
        let bool_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(text_query) as Box<dyn Query>),
            (Occur::Must, Box::new(title_filter) as Box<dyn Query>),
        ]);
        Ok(Box::new(bool_query))
    }

    pub fn search(
        &mut self,
        query: &str,
        books: &Vec<String>,
        limit: u32,
        fuzzy: bool,
    ) -> Result<Vec<SearchResult>> {
        let index = &self.index;
        let schema = &self.schema;
        let query = Self::create_search_query(index, query, books, fuzzy)?;
        let searcher = index.reader()?.searcher();

        let mut results = Vec::<SearchResult>::new();
        let title_field = schema.get_field("title")?;
        let text_field = schema.get_field("text")?;
        let id_field = schema.get_field("id")?;
        let segment_field = schema.get_field("segment")?;
        let is_pdf_field = schema.get_field("isPdf")?;
        let file_path_field = schema.get_field("filePath")?;
        let mut snippet_generator = SnippetGenerator::create(&searcher, &*query, text_field)?;
        snippet_generator.set_max_num_chars(800);

        let top_docs: Vec<DocAddress> = {
            if fuzzy {
                // sort by relevance
                let collector_by_relanace = TopDocs::with_limit(limit as usize);
                let top_docs_by_relevance = searcher.search(&query, &collector_by_relanace)?;
                top_docs_by_relevance
                    .into_iter()
                    .map(|(score, doc_address)| (doc_address))
                    .collect()
            } else {
                // sort by id (ascending)
                let collector_by_id = TopDocs::with_limit(limit as usize)
                    .order_by_fast_field::<u64>("id", Order::Asc);
                let top_docs_by_id = searcher.search(&query, &collector_by_id).unwrap();
                top_docs_by_id
                    .into_iter()
                    .map(|(id, doc_address)| (doc_address))
                    .collect()
            }
        };

        for doc_address in top_docs {
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
                    let mut snippet = snippet_generator.snippet(&text);
                    snippet.set_snippet_prefix_postfix("<font color=red>", "</font>");
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
                    let text = {
                        if fuzzy {
                            text
                        } else {
                            snippet.to_html()
                        }
                    };

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
    pub fn search_stream(
        &mut self,
        query: &str,
        sink: StreamSink<Vec<SearchResult>>,
        books: &Vec<String>,
        limit: u32,
        fuzzy: bool,
    ) -> Result<()> {
        let index = &self.index;
        let schema = &self.schema;
        let query = Self::create_search_query(index, query, books, fuzzy).unwrap();
        let searcher = index.reader().unwrap().searcher();
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit as usize))
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
                    sink.add(results.clone());
                }
                Err(_) => continue,
            }
        }
        Ok(())
    }
}
