use crate::frb_generated::StreamSink;
use anyhow::{Context, Result};
use flutter_rust_bridge::frb;
use levenshtein_automata::{Distance, LevenshteinAutomatonBuilder, DFA, SINK_STATE};
use log::{debug, error, info, warn};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tantivy::collector::{Collector, Count, FacetCollector, SegmentCollector, TopDocs};
use tantivy::directory::MmapDirectory;
use tantivy::index::{SegmentId, SegmentMeta};
use tantivy::indexer::NoMergePolicy;
use tantivy::query::{
    AllQuery, BooleanQuery, BoostQuery, ConstScoreQuery, EmptyQuery, FuzzyTermQuery, Occur,
    PhraseQuery, TermQuery, TermSetQuery,
};
use tantivy::query::{Query, RegexPhraseQuery};
use tantivy::schema::Value;
use tantivy::snippet::SnippetGenerator;
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, TextAnalyzer, TokenStream};
use tantivy::{doc, DocAddress, IndexReader, IndexWriter, Order, ReloadPolicy, Score, Searcher};
use tantivy::{schema::*, Index};
use tantivy::{DocId, SegmentOrdinal, SegmentReader};
use tantivy_fst::Automaton;

use crate::display_highlight;
use crate::gap_phrase::{GapVerifiedPhraseQuery, TermListPhraseQuery};
use crate::hebrew_query;
use crate::hebrew_query::VocalizedFlags;
use crate::hebrew_tokenizer::HebrewTokenizer;
use crate::lexicons::{
    AcronymLexicon, TranslationLexicon, MAX_ACRONYM_EXPANSIONS, MAX_TRANSLATION_EXPANSIONS,
};
use crate::magic::{MagicDictionary, MAX_LEXICAL_FORMS};
use crate::section_scope::{SectionFilteredQuery, SectionIdsCollector};

#[cfg(feature = "semantic-integration")]
use otzaria_semantic_search::api::hybrid_search::{
    OtzariaHybridEngine, SearchRequest as SidecarSearchRequest,
};
#[cfg(feature = "semantic-integration")]
use otzaria_semantic_search::hybrid::coordinator::HybridCoordinator;
#[cfg(feature = "semantic-integration")]
use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
#[cfg(feature = "semantic-integration")]
use otzaria_semantic_search::semantic::types::{
    BookForIndexing as SidecarBookForIndexing, BookLine as SidecarBookLine,
    GroupingMode as SidecarGroupingMode, LexicalCandidate as SidecarLexicalCandidate,
    ResultSource as SidecarResultSource, SearchFilters as SidecarSearchFilters,
    SearchMode as SidecarSearchMode,
};

// ── Public data types ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SearchResult {
    pub title: String,
    pub reference: String,
    pub text: String,
    pub id: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
    /// כמה תוצאות גולמיות מאוחדות בכרטיס הזה (1 = ללא איחוד). ראו
    /// [`ResultGrouping`]: בחיפוש מקובץ התוצאה היא נציג הקבוצה, והמונה
    /// הוא "נמצאו X תוצאות בטווח". כש-`truncated` דלוק המונה הוא תחתית —
    /// קבוצה שפונתה מהתקרה וחזרה מאבדת את חבריה המוקדמים.
    pub merged_count: u32,
    /// שאר חברות הקבוצה (ללא הנציג), עד [`MERGED_SIBLINGS_CAP`] — מיקומים
    /// בלבד, בלי snippet, כדי שהאפליקציה תוכל להציג "הרחב" ולקפוץ אליהן.
    /// `merged_count` עשוי לעלות על `merged.len() + 1` כשהקבוצה גדולה מהתקרה.
    pub merged: Vec<MergedSibling>,
}

/// חברת קבוצה מאוחדת: מיקום בלבד (בלי טקסט/הדגשה) — מספיק כדי להציג
/// שורת-משנה בכרטיס מקובץ ולפתוח את הספר במקום הנכון.
#[derive(Clone)]
pub struct MergedSibling {
    pub title: String,
    pub reference: String,
    pub id: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
}

/// מצב איחוד תוצאות. `None` בפרמטר `grouping` = ההתנהגות הקיימת (שטוח).
///
/// - `SameSection` — כל התוצאות שתחת אותה כותרת באותו ספר (אותו `sectionId`;
///   ב-PDF: אותו עמוד) מתאחדות לכרטיס אחד עם מונה — "נמצאו X תוצאות בטווח".
/// - `IdenticalText` — שורות שגוף הטקסט העברי שלהן זהה (חתימת `lineHash`)
///   מתאחדות גם חוצה-ספרים — אותה משנה במהדורות שונות, מדרש מצוטט וכו'.
///   שורה קצרה מדי לחתימה (ראו [`line_dedup_hash`]) לעולם אינה מתאחדת.
///
/// הנציג של קבוצה הוא התוצאה הטובה ביותר לפי סדר המיון הנוכחי, והקבוצות
/// ממוינות לפי הנציג; `limit`/`offset` נספרים בקבוצות (לא בתוצאות גולמיות).
#[derive(Clone, Copy)]
pub enum ResultGrouping {
    SameSection,
    IdenticalText,
}

pub struct DocumentInput {
    pub id: u64,
    pub title: String,
    pub reference: String,
    pub topics: String,
    pub text: String,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
    /// Book-level **canonical** fingerprint for the `contentHash` column —
    /// [`compute_book_fingerprint`]: the raw text *plus* the metadata baked
    /// into the index (title, topics, catalogue order, generation order,
    /// extra facets). The same value is stamped on every document of a book,
    /// so [`SearchEngine::get_book_fingerprints`] can compare an index
    /// against the current library source, including metadata-only changes.
    /// Never compare it against [`compute_content_fingerprint`] (text only) —
    /// that is `text_hash` below. `None`/`0` means "no fingerprint recorded"
    /// (e.g. PDF books).
    pub content_hash: Option<u64>,
    /// Text-only fingerprint ([`compute_content_fingerprint`] over the raw
    /// book text) for the `textHash` column — unlike `content_hash` it does
    /// not shift when catalogue order or other metadata changes, so
    /// [`SearchEngine::get_book_text_fingerprints`] detects content drift
    /// only. `None`/`0` means "no fingerprint recorded".
    pub text_hash: Option<u64>,
    /// הטקסט המנוקד של השורה (נרמול [`normalize_vocalized_text_for_indexing`])
    /// עבור השדה `textVocalized`. `None` לשורה ללא ניקוד/טעמים — השורה
    /// פשוט לא תשתתף בחיפוש מנוקד. מסלול [`SearchEngine::add_text_book`]
    /// מחשב זאת בעצמו; השדה קיים לצינורות שמוסיפים מסמכים מוכנים.
    pub text_vocalized: Option<String>,
    /// מזהה הסעיף (בלוק הכותרת) של השורה — ראו השדה `sectionId` בסכימה.
    /// `None` = השורה סעיף לעצמה (`id` משמש כמזהה), כך שחיפוש "תחת אותה
    /// כותרת" מתנהג כמו "באותה פסקה" עבור מסמכים שהוזנו בלי מזהה סעיף.
    pub section_id: Option<u64>,
    /// סדר הדור של הספר (נמוך = מוקדם). `None` ממוין לסוף הרשימה.
    pub generation_order: Option<u32>,
    /// נתיבי facet נוספים לצד `topics` — ממדי סינון (מחבר/תקופה/ספר-יסוד),
    /// למשל `"/author/רש\"י"`, `"/era/ראשונים"`, `"/base"`. ראו
    /// [`FACET_DIMENSION_ROOTS`] לסמנטיקת הסינון (OR בתוך ממד, AND בין
    /// ממדים). `None` = אין.
    pub extra_facets: Option<Vec<String>>,
}

/// One extracted PDF page for [`SearchEngine::add_pdf_book`]: the page's
/// display reference (built by the app from the PDF outline), the raw
/// extracted page text, and the zero-based page index (stored as the
/// document `segment`).
pub struct PdfPageInput {
    pub reference: String,
    pub text: String,
    pub page_index: u32,
}

pub struct HighlightConfig {
    pub highlight_prefix: String,
    pub highlight_postfix: String,
    pub max_chars: u32,
}

pub struct SearchPageResult {
    pub total_count: u32,
    pub results: Vec<SearchResult>,
    /// `true` when a broad single-word query overflowed its collection budget
    /// and only the highest-priority term expansions were served, so both
    /// `total_count` and `results` are partial (see
    /// [`SearchEngine::single_regex_term_query`]). The regex, advanced and
    /// *vocalized* exact/fuzzy paths can degrade this way (a vocalized word
    /// materializes a term set like an advanced word); the mark-free
    /// exact/fuzzy paths never do this. A *grouped* search on any path can
    /// also set it when the group cap overflows
    /// ([`GROUP_COLLECTOR_MAX_GROUPS`]) — then `group_count` is a lower
    /// bound while `total_count` stays exact.
    pub truncated: bool,
    /// מספר הקבוצות הכולל כשהחיפוש רץ עם `grouping` — זה המספר שדפדוף
    /// (limit/offset) נספר בו. `None` בחיפוש שטוח. `total_count` נשאר
    /// תמיד ספירת התוצאות הגולמיות.
    pub group_count: Option<u32>,
}

/// One event of a combined stream search (`search_*_stream_with_counts`).
///
/// The first event carries the counts computed in the *same* index pass as
/// the ranked results (`total_count` + `book_counts`, with empty `results`);
/// every following event is a snippet-built results chunk (`None` counts).
/// One user search previously cost three full query executions — stream,
/// total count, and count-by-book — this collapses them into one.
pub struct SearchStreamUpdate {
    /// Full hit count of the query; `Some` only on the first event.
    pub total_count: Option<u32>,
    /// Live-document count per distinct `filePath`; `Some` only on the first
    /// event. Sums to `total_count`.
    pub book_counts: Option<HashMap<String, u32>>,
    /// The results chunk (empty on the first, counts-bearing event).
    pub results: Vec<SearchResult>,
    /// `true` when a broad single-word query overflowed its collection budget
    /// and only the highest-priority term expansions were served, so both the
    /// counts and the results are partial (see [`SearchEngine::single_regex_term_query`]).
    /// Meaningful only on the first, counts-bearing event; always `false` on
    /// result chunks and on the mark-free exact/fuzzy paths, which never
    /// degrade this way (the vocalized exact/fuzzy paths can, like the
    /// advanced path). A *grouped* search on any path can also set it when
    /// the group cap overflows ([`GROUP_COLLECTOR_MAX_GROUPS`]) — then
    /// `group_count` is a lower bound while `total_count`/`book_counts`
    /// stay exact. The UI surfaces this as a "results may be partial —
    /// narrow the search" warning.
    pub truncated: bool,
    /// מספר הקבוצות הכולל כשהחיפוש רץ עם `grouping`; `Some` רק באירוע
    /// הראשון (נושא-הספירות), כמו `total_count`. `total_count` ו-
    /// `book_counts` נשארים ספירות גולמיות גם בחיפוש מקובץ.
    pub group_count: Option<u32>,
}

pub struct FacetCount {
    pub path: String,
    pub count: u64,
}

/// A total hit count paired with the single-word truncation flag — the
/// status-bearing return of [`SearchEngine::count_with_status`] and its
/// advanced/exact/fuzzy variants. `truncated` carries the same meaning as
/// [`SearchStreamUpdate::truncated`]: `true` when a broad single-word query
/// overflowed its collection budget, so `count` undercounts the true total.
pub struct CountResult {
    pub count: u32,
    pub truncated: bool,
}

/// Per-`filePath` live-document counts paired with the truncation flag — the
/// status-bearing return of [`SearchEngine::count_by_book_with_status`] and
/// its advanced/exact/fuzzy variants. When `truncated`, the per-book counts
/// are partial.
pub struct BookCountResult {
    pub counts: HashMap<String, u32>,
    pub truncated: bool,
}

/// Per-child facet counts paired with the truncation flag — the status-bearing
/// return of [`SearchEngine::get_facet_counts_with_status`] and its
/// advanced/exact/fuzzy variants. When `truncated`, the facet counts are
/// partial.
pub struct FacetCountsResult {
    pub counts: Vec<FacetCount>,
    pub truncated: bool,
}

#[derive(Clone)]
pub struct IndexCompatibility {
    pub compatible: bool,
    pub status: String,
    pub found_schema_version: Option<u32>,
    pub required_schema_version: u32,
    pub engine_version: String,
    pub metadata_path: String,
    pub reason: Option<String>,
}

// ── Semantic sidecar FFI API ─────────────────────────────────────────────────
//
// The lexical query mode and the retrieval mode deliberately have separate
// types. The former says how Tantivy interprets the text; the latter says which
// retrieval paths the sidecar may use. Keeping that boundary here prevents a
// caller from accidentally treating e.g. a fuzzy query as a semantic mode.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticLexicalMode {
    Exact,
    Fuzzy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticRetrievalMode {
    Hybrid,
    SemanticOnly,
    LexicalOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticGroupingMode {
    SameSection,
    IdenticalText,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticExecutedMode {
    Disabled,
    Hybrid,
    SemanticOnly,
    LexicalOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticResultSource {
    Lexical,
    Semantic,
    Both,
}

/// Configuration needed to open the semantic sidecar. The model itself is
/// loaded lazily when indexing begins, so configuration is cheap; searches
/// report a degraded state until indexing has loaded the model and produced
/// vectors, instead of making the lexical engine unusable.
pub struct SemanticConfigInput {
    pub root_dir: String,
    pub model_path: String,
    pub model_id: String,
    pub embedding_dim: u32,
}

/// A serializable, feature-independent projection of sidecar status. It is
/// intentionally available without the `semantic` Cargo feature so Dart can
/// render an explicit Disabled state rather than silently falling back.
pub struct SemanticStatus {
    pub enabled: bool,
    pub available: bool,
    pub model_loaded: bool,
    pub indexed_book_count: u32,
    pub vector_count: u32,
    pub model_id: String,
    pub embedding_dim: u32,
    pub embedding_backend: Option<String>,
    pub vector_backend: String,
    pub vectors_persisted: bool,
    pub needs_full_reindex: Option<String>,
    pub last_error: Option<String>,
}

pub struct SemanticBookLineInput {
    pub line_id: u64,
    pub section_id: u64,
    pub text: String,
    pub line_hash: u64,
    pub reference: String,
    pub segment: u64,
}

pub struct SemanticBookInput {
    pub source_book_key: String,
    pub title: String,
    /// The lexical engine's book content hash, or a caller-provided canonical
    /// source fingerprint for PDFs. Zero means it cannot be verified.
    pub content_fingerprint: u64,
    pub is_pdf: bool,
    pub topics: String,
    pub extra_facets: Vec<String>,
    pub lines: Vec<SemanticBookLineInput>,
}

pub struct SemanticIndexingSummary {
    pub enabled: bool,
    pub books_indexed: u32,
    pub books_skipped: u32,
    pub books_empty: u32,
    pub chunks_written: u32,
}

pub struct SemanticIndexDiff {
    pub enabled: bool,
    pub new_books: Vec<String>,
    pub changed_books: Vec<String>,
    pub unverifiable_books: Vec<String>,
    pub removed_books: Vec<String>,
    pub model_mismatch: bool,
    pub chunking_mismatch: bool,
    pub normalization_mismatch: bool,
}

pub struct SemanticResetResult {
    pub enabled: bool,
    pub vectors_removed: u32,
}

pub struct SemanticRemoveResult {
    pub enabled: bool,
    pub vectors_removed: u32,
}

pub struct SemanticSearchResult {
    pub title: String,
    pub reference: String,
    /// The display string, in the same format every other search API in this
    /// engine returns: HTML-escaped and painted with `HighlightConfig`'s
    /// prefix/postfix where the lexical query matched. It is never a raw
    /// unbounded line, on either the sidecar or the fallback path — the app's
    /// snippet parser can treat both alike. `max_chars` bounds how much of the
    /// line is shown; markup and escaping are added on top of that.
    ///
    /// The sidecar path paints against the mark-free stored `text` field,
    /// because that is the copy the sidecar indexes and hydration reads. A
    /// vocalized query therefore selects documents with marks but still shows
    /// (and highlights) the mark-free line; the lexical fallback shows the
    /// vocalized copy. Use [`SearchEngine::get_document_by_id`] when the full,
    /// unabridged line is needed.
    pub snippet_html: String,
    /// Whether `snippet_html` carries highlight markup. False for a purely
    /// semantic hit, whose line matched no query term — the UI can then avoid
    /// promising the user a lexical match that is not there.
    pub is_highlighted: bool,
    pub id: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
    pub merged_count: u32,
    pub merged: Vec<MergedSibling>,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub fused_score: f32,
    pub source: SemanticResultSource,
    /// False after successful Tantivy hydration of a semantic-only item.
    pub needs_hydration: bool,
}

/// Result envelope for semantic/hybrid searches. `lexical_total_count` is the
/// truthful corpus-wide Tantivy count; `total_count` is the sidecar candidate
/// fusion count and must not be presented as a corpus-wide semantic total.
pub struct SemanticSearchResponse {
    pub results: Vec<SemanticSearchResult>,
    pub total_count: u32,
    pub lexical_total_count: u32,
    pub group_count: Option<u32>,
    /// Whether `total_count`/`group_count` are exact. Sidecar-backed searches
    /// report candidate-window counts, not a corpus-wide semantic total, and
    /// may also exclude stale records during Tantivy hydration.
    /// `lexical_total_count` uses `truncated` as its accuracy signal instead.
    pub counts_are_exact: bool,
    pub requested_mode: SemanticRetrievalMode,
    pub executed_mode: SemanticExecutedMode,
    pub semantic_available: bool,
    pub fallback_reason: Option<String>,
    pub latency_ms: u64,
    /// The sidecar input window hit its hard memory-safety ceiling. This is
    /// separate from `truncated`, which belongs to lexical term expansion.
    pub candidate_window_truncated: bool,
    pub truncated: bool,
}

/// The four inputs that decide which vectors a sidecar session holds. Kept
/// beside the open engine so [`SearchEngine::configure_semantic`] can tell a
/// harmless repeat call from a real change of model or library root — see that
/// method for why the difference matters.
#[cfg(feature = "semantic-integration")]
#[derive(Clone, PartialEq, Eq)]
struct SemanticConfigKey {
    root_dir: PathBuf,
    model_path: PathBuf,
    model_id: String,
    embedding_dim: u32,
}

#[cfg(feature = "semantic-integration")]
impl SemanticConfigKey {
    fn from_input(config: &SemanticConfigInput) -> Self {
        Self {
            root_dir: PathBuf::from(&config.root_dir),
            model_path: PathBuf::from(&config.model_path),
            model_id: config.model_id.clone(),
            embedding_dim: config.embedding_dim,
        }
    }

    /// Names the fields that differ, so a refused reconfiguration says which
    /// input changed instead of only that something did.
    fn changed_fields(&self, other: &Self) -> String {
        let mut changed = Vec::new();
        if self.root_dir != other.root_dir {
            changed.push("root_dir");
        }
        if self.model_path != other.model_path {
            changed.push("model_path");
        }
        if self.model_id != other.model_id {
            changed.push("model_id");
        }
        if self.embedding_dim != other.embedding_dim {
            changed.push("embedding_dim");
        }
        changed.join(", ")
    }
}

/// An open sidecar session: the engine plus the configuration that produced it.
#[cfg(feature = "semantic-integration")]
struct ConfiguredSemantic {
    engine: OtzariaHybridEngine,
    config: SemanticConfigKey,
}

#[cfg(feature = "semantic-integration")]
type SemanticRuntime = Option<ConfiguredSemantic>;
#[cfg(not(feature = "semantic-integration"))]
type SemanticRuntime = ();

/// What the lexical half of a sidecar search produces: the scored candidates
/// fusion consumes, the corpus-wide count for the response envelope, and the
/// inputs needed to paint the *final page* the way the lexical API paints its
/// own results.
#[cfg(feature = "semantic-integration")]
struct SemanticLexicalPhase {
    candidates: Vec<SidecarLexicalCandidate>,
    total_count: u32,
    truncated: bool,
    /// `None` in `SemanticOnly`, where no lexical query is executed and every
    /// result therefore gets an unhighlighted (but still bounded) snippet.
    highlight: Option<SemanticHighlight>,
}

/// A lexical query kept alive past its own execution so snippets can be built
/// after fusion and pagination rather than for the whole candidate window.
#[cfg(feature = "semantic-integration")]
struct SemanticHighlight {
    /// Drives both fragment selection and term painting.
    query: Box<dyn Query>,
    phrase: Option<PhraseHighlight>,
}

/// Paints one page of sidecar results. Built once per page — creating the
/// generator resolves the doc-frequency of every highlight term — and only when
/// there is a page to paint.
#[cfg(feature = "semantic-integration")]
struct SemanticSnippetPainter {
    searcher: Searcher,
    generator: SnippetGenerator,
    phrase: Option<PhraseHighlight>,
    hl: HighlightConfig,
}

#[cfg(feature = "semantic-integration")]
impl SemanticSnippetPainter {
    /// Returns the display markup for `text` and whether it carries highlights.
    ///
    /// Mirrors [`SearchEngine::build_results_with_generator`]: tantivy's
    /// term-based highlighter picks the fragment, and a phrase plan re-derives
    /// the painting so only complete in-order occurrences stay painted. A
    /// fragment only exists when at least one query term matched, so an empty
    /// result means this line is a purely semantic hit — it then falls back to a
    /// bounded escaped snippet instead of an unbounded raw line.
    ///
    /// `lexically_confirmed` is what makes the phrase fallback honest. When the
    /// filter finds no complete in-order occurrence, the lexical API paints the
    /// individual terms instead — sound there, because Tantivy already proved the
    /// document satisfies the phrase query, so the fragment merely failed to
    /// contain a whole occurrence. A purely semantic candidate never passed that
    /// query: painting its scattered words would assert a phrase match that does
    /// not exist. Such a result is left unpainted instead.
    fn paint(&self, text: &str, lexically_confirmed: bool) -> (String, bool) {
        let mut snippet = self.generator.snippet(text);
        snippet.set_snippet_prefix_postfix(&self.hl.highlight_prefix, &self.hl.highlight_postfix);
        let html = match self.phrase.as_ref() {
            Some(phrase) => SearchEngine::phrase_filtered_snippet_html(
                &self.searcher,
                snippet.fragment(),
                phrase,
                &self.hl,
            )
            .or_else(|| lexically_confirmed.then(|| snippet.to_html())),
            // No phrase constraint: every occurrence of every query word is a
            // real match of that word, whichever retrieval path found the line,
            // so term painting states nothing untrue.
            None => Some(snippet.to_html()),
        };
        match html {
            Some(html) if !html.is_empty() => (html, true),
            _ => (bounded_plain_snippet(text, self.hl.max_chars), false),
        }
    }
}

/// A display-safe stand-in for a snippet: a bounded, HTML-escaped prefix, so a
/// line that cannot be painted still crosses FFI as markup instead of a raw full
/// line.
///
/// `max_chars` bounds the *source* line, in bytes, matching tantivy's own
/// `max_num_chars`; the cut lands on a UTF-8 char boundary so multi-byte Hebrew
/// is never split. Escaping happens after the cut, so the returned string can be
/// longer than the budget when the line is dense in `&` or `<` — the bound is on
/// how much text is shown, not on the byte length of its encoding.
fn bounded_plain_snippet(text: &str, max_chars: u32) -> String {
    let budget = max_chars as usize;
    if text.len() <= budget {
        return htmlescape::encode_minimal(text);
    }
    let mut end = budget;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut html = htmlescape::encode_minimal(&text[..end]);
    html.push('…');
    html
}

/// טווח הקרבה הנדרש בין מילות שאילתה מרובת-מילים במסלול המתקדם.
pub enum SearchScope {
    /// ההתנהגות הקיימת: המילים מופיעות לפי סדר השאילתה, עם מגבלת
    /// מילים-ביניים לכל זוג סמוך (`distance` / `custom_spacing`).
    WordDistance,
    /// כל המילים באותה פסקה (מסמך אינדקס אחד = שורת ספר), בכל סדר ובכל
    /// מרחק. `distance`/`custom_spacing` אינם רלוונטיים במצב זה.
    SameParagraph,
    /// כל המילים תחת אותה כותרת (אותו בלוק `reference` — סעיף/פרק), גם
    /// כשהן פזורות על פני שורות שונות. התוצאות הן השורות שבתוך סעיף
    /// חותך שמכילות מילה מהשאילתה.
    SameSection,
}

/// כמה ממילות שאילתה מרובת-מילים חייבות להופיע בתוצאה (המסלול המתקדם).
///
/// בכל מצב שאינו [`WordMatchMode::All`] דרישת הסדר והמרחק חסרת משמעות —
/// מילים עשויות להיות חסרות — ולכן ההתאמה נעשית ברזולוציית ה-scope בלבד:
/// `WordDistance` מתנהג כ-`SameParagraph`, ו-`SameSection` דורש שהסעיף
/// יכיל לפחות את מספר המילים הנדרש.
pub enum WordMatchMode {
    /// ההתנהגות הקיימת: כל המילים חובה.
    All,
    /// די במילה אחת ממילות השאילתה.
    AnyWord,
    /// רוב המילים: יותר ממחצית (`n/2 + 1`).
    MostWords,
    /// לפחות `word_match_count` מילים (הפרמטר הנלווה); נחתך ל-`[1, n]`.
    AtLeast,
}

/// הצורה הפנימית של צמד הפרמטרים `word_match_mode`/`word_match_count`.
/// (מופרד מ-[`WordMatchMode`] כי enum נושא-נתונים בגשר היה גורר תלות freezed.)
enum WordMatch {
    All,
    AtLeast(u32),
    Most,
}

impl WordMatch {
    fn from_api(mode: Option<WordMatchMode>, count: Option<u32>) -> Self {
        match mode.unwrap_or(WordMatchMode::All) {
            WordMatchMode::All => WordMatch::All,
            WordMatchMode::AnyWord => WordMatch::AtLeast(1),
            WordMatchMode::MostWords => WordMatch::Most,
            WordMatchMode::AtLeast => WordMatch::AtLeast(count.unwrap_or(1).max(1)),
        }
    }

    /// מספר המילים המינימלי הנדרש עבור שאילתה בת `word_count` מילים.
    fn min_words(&self, word_count: usize) -> usize {
        let min = match self {
            WordMatch::All => word_count,
            WordMatch::AtLeast(count) => *count as usize,
            WordMatch::Most => word_count / 2 + 1,
        };
        min.min(word_count)
    }
}

pub enum ResultsOrder {
    Catalogue,
    Relevance,
    Generation,
}

/// Per-word index-term sets plus the intermediate-word allowance, used by the
/// snippet renderer to keep only highlights that form an in-order *phrase*
/// occurrence.
///
/// tantivy's `SnippetGenerator` highlights term-by-term: it paints every
/// occurrence of every query term with no positional constraint. For the
/// multi-word phrase paths (exact `PhraseQuery`, advanced/lexical-fuzzy
/// `RegexPhraseQuery`) that over-paints — a lone "משה" lights up even when the
/// search matched only the adjacent phrase "משה ואהרן". This filter drops such
/// stray highlights so the snippet paints exactly what the search matched.
///
/// `gaps[i]` is the maximum number of intermediate index tokens allowed
/// between query words `i` and `i+1` — the same per-pair intermediate-word
/// model `display_highlight` and the engine's phrase verification use, so the
/// results snippet, the search, and the book agree.
struct PhraseHighlight {
    /// One index-term set per query word, in query (= phrase) order.
    per_word_terms: Vec<HashSet<String>>,
    /// Per adjacent pair; length `per_word_terms.len() - 1`.
    gaps: Vec<u32>,
    /// The analyzer that tokenizes snippet fragments for re-highlighting —
    /// must match the field the terms came from (`"hebrew"` for `text`,
    /// `"hebrew_vocalized"` for `textVocalized`).
    analyzer: &'static str,
}

/// What a highlight-query builder resolves to: the flat term query that drives
/// tantivy's fragment selection (and, absent a phrase filter, its
/// highlighting), plus the optional phrase constraint above.
struct HighlightPlan {
    /// `None` falls back to the main search query, which already exposes its
    /// terms to `SnippetGenerator` when it is a Term/Phrase/TermSet query.
    query: Option<Box<dyn Query>>,
    phrase: Option<PhraseHighlight>,
}

impl HighlightPlan {
    /// No highlight query and no phrase filter — the snippet generator falls
    /// back to the main query's own terms (single-word/plain paths).
    fn none() -> Self {
        HighlightPlan {
            query: None,
            phrase: None,
        }
    }
}

/// What [`SearchEngine::build_advanced_query`] returns: the executable query,
/// one joined regex pattern per word (for highlight-term materialization), the
/// resolved per-pair gap allowances (fold `custom_spacing`/`distance` in — the
/// phrase highlight filter's gap allowances), and whether single-word
/// collection truncated. The fifth element carries the acronym-expansion
/// alternatives ("ראשי תיבות") as per-word literal-pattern lists — the
/// highlight builders materialize their terms so a document matched through
/// an expansion (רמב"ם → "רבי משה בן מיימון") still gets its snippet painted.
type AdvancedQueryBuild = (
    Box<dyn Query>,
    Vec<String>,
    Vec<u32>,
    bool,
    Vec<Vec<String>>,
);

/// Regex patterns for highlighting query matches in *displayed* book text
/// (which, unlike index terms, still carries nikud and HTML). All patterns
/// are ECMAScript-dialect strings; the Dart layer compiles them with
/// `RegExp(pattern, caseSensitive: false)` and performs no pattern
/// construction of its own.
pub struct HighlightPattern {
    /// One regex matching the full query phrase (words + separators).
    pub combined_pattern: String,
    /// Per-word regex, used to locate each word inside a combined match.
    pub word_patterns: Vec<String>,
    /// Per-word: `true` when the word has no morphological expansion option,
    /// so the UI may require token boundaries around its match.
    pub word_boundary_eligible: Vec<bool>,
}

// ── SearchEngine ───────────────────────────────────────────────────────────────

// `Index::writer(budget)` splits the budget across indexing threads and
// requires ≥15MB per thread, so the budget effectively picks the thread
// count: 50MB caps tantivy at 3 threads (≈16MB arenas — frequent flushes,
// many small segments), while 300MB lets it use all 8 (37.5MB arenas). The
// budget is only consumed while indexing is active; mobile keeps the small
// footprint.
#[cfg(any(target_os = "android", target_os = "ios"))]
const DEFAULT_WRITER_HEAP_SIZE: usize = 50_000_000;
#[cfg(not(any(target_os = "android", target_os = "ios")))]
const DEFAULT_WRITER_HEAP_SIZE: usize = 300_000_000;
// Economy indexing (see `set_economy_indexing`): the mobile budget — 3
// threads with small arenas — applied on demand on desktop too.
const ECONOMY_WRITER_HEAP_SIZE: usize = 50_000_000;
/// Upper bound on searchable segments left behind by `optimize`.
const MAX_SEGMENTS_AFTER_OPTIMIZE: usize = 8;
/// A segment whose deleted-doc share exceeds this is compacted regardless of size.
const OPTIMIZE_COMPACT_DELETE_RATIO: f64 = 0.3;
const INDEX_METADATA_FILE_NAME: &str = "otzaria_index_meta.json";
const INDEX_FORMAT: &str = "otzaria-search-index";
// גרסה 3 (טרם פורסמה): המעבר ל-HebrewTokenizer (גרשיים/גרש נשמרים
// בטוקנים) משנה את מילון הטרמים, הוסר ה-fast field מ-`text` (עותק columnar
// של כל הקורפוס שאיש לא קרא), ונוסף השדה `textVocalized` (אינדקס + אחסון
// של שורות מנוקדות, טוקנייזר ששומר ניקוד/טעמים) עבור חיפוש מנוקד —
// הסכימה בדיסק שונה מגרסה 2, אינדקסים ישנים חייבים בנייה מחדש.
//
// שים לב: נוסף השדה `sectionId` (FAST) עבור חיפוש "תחת אותה כותרת" —
// שינוי סכימה שמחייב העלאת גרסה לפני פרסום (בדיקת התאימות משווה גם את
// סכימת ה-tantivy בפועל, כך שאינדקס ישן ידווח rebuild_required גם בלעדיה).
//
// שים לב: הטמעת הטוקן-התאום נטול-הגרשיים (emit_quote_free בטוקנייזרים)
// משנה את *תוכן* מילון הטרמים בלי לשנות את סכימת ה-tantivy — בדיקת
// ההתאימות לא תתפוס זאת מעצמה, ולכן חובה להעלות את הגרסה לפני פרסום
// כדי שאינדקסים ישנים ייבנו מחדש (בלעדיהם חיפוש `רמבם` לא ימצא `רמב"ם`
// ו"התעלם מגרשיים" לא יעבוד על ספרים שאונדקסו קודם).
//
// שים לב: נוספו השדה `lineHash` (FAST, חתימת דה-דופ לשורה) ו-facets
// ממדיים (author/era/base על שדה topics) — שינויי סכימה/תוכן שמחייבים
// העלאת גרסה לפני פרסום (את lineHash בדיקת ההתאימות תתפוס גם לבדה;
// את ה-facets הממדיים — לא, כמו הטוקן נטול-הגרשיים לעיל).
//
// v4: נוסף השדה `textHash` (FAST) — חתימת טקסט-בלבד לצד החתימה הקנונית,
// כדי שאימות דריפט תוכן לא ייפסל משינויי metadata (סדר קטלוגי וכו').
pub(crate) const INDEX_SCHEMA_VERSION: u32 = 4;
const TANTIVY_INDEX_VERSION: &str = "0.26.1";

/// תקרת אורך טוקן (בבייטים של UTF-8) לכל האנליזטורים — אינדוקס ושאילתה
/// כאחד. 128 בייט ≈ 64 אותיות עבריות: פי כמה מכל מילה לגיטימית (כולל
/// גרשיים וניקוד), ומתחת לכל ריצת-זבל אמיתית (base64 שזלג לקורפוס).
const MAX_TOKEN_BYTES: usize = 128;
const DEFAULT_GENERATION_ORDER: u32 = 5;
const GENERATION_SORT_SHIFT: u32 = 56;
const GENERATION_SORT_ID_MASK: u64 = (1u64 << GENERATION_SORT_SHIFT) - 1;

/// Upper bound on distinct dictionary terms collected for highlighting an
/// advanced (regex) query. Bounds work when a pattern (e.g. partial match)
/// expands very widely; far more matches than a snippet could ever show.
/// Scaled ×4 with the search-side expansion ceilings (parity: a document
/// found via a wide expansion should still highlight its variant); the
/// display char budget bounds the final pattern size regardless.
const MAX_HIGHLIGHT_TERMS: usize = 2_048;
const MAX_LEXICAL_PHRASE_TERMS_PER_TOKEN: usize = 256;
const LEXICAL_FUZZY_PHRASE_SLOP: u32 = 1;

// Relevance weights for the approximate (`fuzzy`) path. `FuzzyTermQuery` and
// `TermSetQuery` are automaton queries that score a flat 1.0 (`ConstScorer`),
// so without boosting every approximate hit ties and `order_by_score` produces
// no visible ordering. These tiers make `ResultsOrder::Relevance` meaningful:
// an exact-token hit outranks a dictionary-morphology relative, which outranks
// a bare edit-distance match.
//
// The exact tier is built from TWO clauses that sum (`BooleanQuery` sums
// `Should` scores): a `ConstScoreQuery` floor (`FUZZY_BOOST_EXACT`) plus a small
// BM25 `TermQuery` (`FUZZY_BOOST_EXACT_REL`) for intra-exact ordering. A plain
// boosted `TermQuery` would NOT suffice: BM25 `idf` collapses to ~0 for a term
// present in almost every document (`ln(1 + 0.5/(doc_freq+0.5))`), so a purely
// multiplicative boost could sink an exact hit below the flat lexical tier. The
// constant floor guarantees exact > lexical regardless of `doc_freq`, while the
// BM25 add-on still ranks rarer exact matches first within the top tier.
//
// These layers are added ONLY for `ResultsOrder::Relevance` (the `rank` flag).
// Count/catalogue paths build the bare recall query so they pay nothing for
// ranking they never use. Recall is unchanged either way (the exact term is a
// subset of the fuzzy automaton match), and exact/advanced never use these.
const FUZZY_BOOST_EXACT: Score = 1000.0;
const FUZZY_BOOST_EXACT_REL: Score = 1.0;
const FUZZY_BOOST_LEXICAL: Score = 30.0;
const FUZZY_BOOST_FUZZY: Score = 1.0;

/// The schema fields resolved together by [`SearchEngine::all_fields`]:
/// `(title, reference, text, id, segment, isPdf, filePath, topics,
/// contentHash, textHash, textVocalized, sectionId, generationSort,
/// lineHash)`.
type SchemaFields = (
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
);

/// The high half of every document id in one book: `(catalogue_order + 1) << 32`.
///
/// Refuses the last catalogue position rather than computing it. `u32::MAX + 1` is `2^32`,
/// and shifting that left by 32 overflows a `u64` — wrapping to a base of **zero** in
/// release, which is not a catalogue position at all: every book at position 0 would share
/// its prefix with the raw ordinals, and `semantic_corpus` would refuse the whole index for
/// an id scheme it could not order. One unusable position out of four billion, named here
/// instead of discovered there.
fn catalogue_id_base(catalogue_order: u32) -> Result<u64> {
    if catalogue_order == u32::MAX {
        anyhow::bail!(
            "catalogue_order {catalogue_order} is the one value that cannot form a document \
             id: (catalogue_order + 1) << 32 overflows u64"
        );
    }
    Ok((u64::from(catalogue_order) + 1) << 32)
}

fn generation_sort_key(generation_order: u32, id: u64) -> u64 {
    (u64::from(generation_order.min(255)) << GENERATION_SORT_SHIFT) | (id & GENERATION_SORT_ID_MASK)
}

#[derive(Serialize, Deserialize)]
struct IndexMetadata {
    format: String,
    schema_version: u32,
    engine_version: String,
    tantivy_version: String,
    created_at_unix_seconds: u64,
}

/// Deliberately **not** `#[frb(sync)]` — this reads the index metadata file, and a
/// synchronous binding blocks the calling Dart isolate on disk I/O.
pub fn check_index_compatibility(path: String) -> IndexCompatibility {
    check_index_compatibility_path(Path::new(&path))
}

/// Builds display-highlight regex patterns for a search query, so the app can
/// mark matches inside an opened book exactly the way the engine matched them.
///
/// Pure string computation (no index access) — safe to call synchronously.
/// Parameters mirror [`SearchEngine::search_advanced`]: `distance` is the
/// default intermediate-word allowance, `custom_spacing` is keyed
/// `"i-(i+1)"`, `alternative_words` is keyed by word position, and
/// `search_options` is keyed `"{word}_{index}"` using the same tokenization
/// as engine queries.
///
/// Returns `None` when the query contains no highlightable words.
#[frb(sync)]
pub fn generate_highlight_pattern(
    query: String,
    distance: u32,
    custom_spacing: HashMap<String, String>,
    alternative_words: HashMap<u32, Vec<String>>,
    search_options: HashMap<String, HashMap<String, bool>>,
) -> Option<HighlightPattern> {
    display_highlight::build_display_highlight(
        &query,
        distance,
        &custom_spacing,
        &alternative_words,
        &search_options,
    )
    .map(|hl| HighlightPattern {
        combined_pattern: hl.combined_pattern,
        word_patterns: hl.word_patterns,
        word_boundary_eligible: hl.word_boundary_eligible,
    })
}

/// Builds the regex for highlighting *literal* in-book search matches (the
/// simple/exact mode that scans an open book locally): the phrase as typed,
/// whitespace-joined, nikud-tolerant, geresh/gershayim matching both ASCII and
/// Hebrew forms, with word-boundary lookarounds. The Dart side compiles the
/// returned string with `RegExp(pattern, caseSensitive: false, unicode: true)`
/// and performs no pattern construction of its own.
///
/// Pure string computation — safe to call synchronously. Returns `None` for a
/// whitespace-only query.
#[frb(sync)]
pub fn generate_literal_highlight_pattern(query: String) -> Option<String> {
    display_highlight::build_literal_pattern(&query)
}

/// Normalises a search query exactly like the engine does internally, so the
/// app can build option keys / UI state from the same tokens the engine sees.
///
/// `״→"`, `׳→'`, `־`/`-`→space; strips `,;!?:*()[]{}^$|\+.~\``; collapses
/// whitespace; trims. Pure string computation — safe to call synchronously.
/// This is the single source of truth; the Dart `SearchQueryBuilder.sanitizeQuery`
/// delegates here so index-time and query-time normalisation cannot drift apart.
#[frb(sync)]
pub fn sanitize_query(query: String) -> String {
    hebrew_query::sanitize_query(&query)
}

/// Splits a query into word tokens the same way the engine tokenizes the
/// indexed `text` field (see [`generate_highlight_pattern`] for the key format
/// that consumes these). A `"` or `'` *between* word characters stays inside
/// the token (`רמב"ם`, `ד'אש`), and a trailing `'` is absorbed (`תוס'`); a
/// quote at a word edge separates. See [`hebrew_query::split_query_words`] for
/// the exact rules.
///
/// Pure string computation — safe to call synchronously. Single source of
/// truth for `SearchQueryBuilder.splitQueryWords`.
#[frb(sync)]
pub fn split_query_words(query: String) -> Vec<String> {
    hebrew_query::split_query_words(&query)
}

/// Normalises a text-book line for indexing exactly the way the engine expects
/// stored text to look: strip HTML, decompose presentation forms and strip
/// nikud/cantillation — keeping punctuation, which search results display.
/// Single source of truth for the Dart
/// `IndexingDocumentBuilder.normalizeTextForIndexing`.
///
/// Pure string computation — safe to call synchronously (including from the
/// indexing isolate).
#[frb(sync)]
pub fn normalize_text_for_indexing(input: String) -> String {
    hebrew_query::normalize_text_for_indexing(&input)
}

/// Like [`normalize_text_for_indexing`] but for PDF page text: also drops bidi
/// and zero-width invisibles and collapses whitespace first. Single source of
/// truth for the Dart `IndexingDocumentBuilder.normalizePdfTextForIndexing`.
#[frb(sync)]
pub fn normalize_pdf_text_for_indexing(input: String) -> String {
    hebrew_query::normalize_pdf_text_for_indexing(&input)
}

/// Batch form of [`normalize_text_for_indexing`]: one FFI round-trip per line
/// *batch* instead of one per line. The per-call bridge overhead (string
/// encode/decode + call dispatch) dominates the normalisation itself for
/// short lines, and a full library is millions of lines — the indexing
/// isolate should always prefer this over the single-line form.
#[frb(sync)]
pub fn normalize_texts_for_indexing(inputs: Vec<String>) -> Vec<String> {
    inputs
        .iter()
        .map(|s| hebrew_query::normalize_text_for_indexing(s))
        .collect()
}

/// A PDF line prepared for indexing: the normalised text together with its
/// garbage verdict, so the batch API answers both questions the indexing
/// isolate asks per line in one round-trip.
pub struct PdfIndexLine {
    pub text: String,
    pub is_garbage: bool,
}

/// Batch form of [`normalize_pdf_text_for_indexing`] +
/// [`is_probably_garbage_pdf_text`]: normalises each line and evaluates the
/// garbage heuristic on the result — replacing the two-FFI-calls-per-line
/// pattern with one call per batch.
#[frb(sync)]
pub fn normalize_pdf_texts_for_indexing(inputs: Vec<String>) -> Vec<PdfIndexLine> {
    inputs
        .iter()
        .map(|s| {
            let text = hebrew_query::normalize_pdf_text_for_indexing(s);
            let is_garbage = hebrew_query::is_probably_garbage_pdf_text(&text);
            PdfIndexLine { text, is_garbage }
        })
        .collect()
}

/// Whether a normalised PDF page looks like garbage (OCR noise) and should be
/// skipped. Single source of truth for the Dart
/// `IndexingDocumentBuilder.isProbablyGarbagePdfText`.
#[frb(sync)]
pub fn is_probably_garbage_pdf_text(normalized_text: String) -> bool {
    hebrew_query::is_probably_garbage_pdf_text(&normalized_text)
}

/// 64-bit content fingerprint (FNV-1a over UTF-8 bytes) of a book's raw
/// source text. The whole-book indexing paths ([`SearchEngine::add_text_book`])
/// stamp it on the `textHash` column; recompute it from the current library
/// source and compare against [`SearchEngine::get_book_text_fingerprints`]
/// to detect books whose *content* changed without reindexing everything.
///
/// שימו לב: [`SearchEngine::get_book_fingerprints`] מחזיר את החתימה
/// **הקנונית** [`compute_book_fingerprint`], הכוללת גם metadata (סדר קטלוגי
/// וכו') — השוואה שלה מול הפונקציה הזו (טקסט בלבד) תזהה כל ספר כ"השתנה".
///
/// Never returns 0 — that value is reserved for "no fingerprint recorded".
/// Deliberately hashes the *raw* text (before normalization/tokenization) so
/// the fingerprint does not shift when text-processing internals change.
///
/// Pure string computation — safe to call synchronously (including from the
/// indexing isolate).
#[frb(sync)]
pub fn compute_content_fingerprint(text: String) -> u64 {
    content_fingerprint(&text)
}

/// [`compute_content_fingerprint`] על bytes גולמיים של UTF-8, אסינכרוני —
/// גיבוב ספר שלם אסור שירוץ על ה-UI isolate של הקורא. UTF-8 לא-תקין
/// מוחלף (lossy), בדיוק כמו במסלולי האינדוקס של הספר השלם, כך שהתוצאה
/// שווה לגיבוב הטקסט המפוענח שנחתם בעמודת `textHash`.
pub fn compute_content_fingerprint_bytes(text: Vec<u8>) -> u64 {
    let text = match String::from_utf8_lossy(&text) {
        std::borrow::Cow::Borrowed(_) => {
            // UTF-8 תקין — נטילת בעלות בלי העתקה נוספת.
            unsafe { String::from_utf8_unchecked(text) }
        }
        std::borrow::Cow::Owned(fixed) => fixed,
    };
    content_fingerprint(&text)
}

fn content_fingerprint(text: &str) -> u64 {
    let mut fnv = Fnv::new();
    fnv.feed(text.as_bytes());
    fnv.finish()
}

/// חתימת האינדוקס הקנונית לספר טקסט: הטקסט הגולמי + כל ה-metadata שמוטבע
/// באינדקס — כותרת, נתיב קטגוריה, סדר קטלוגי, סדר דורות וממדי הסינון.
/// שינוי בכל אחד מהם, גם ללא שינוי טקסט, משנה את החתימה — כך שהשוואה מול
/// [`SearchEngine::get_book_fingerprints`] מזהה גם ספר שרק ה-metadata שלו
/// התעדכן (אחרת האינדקס נשאר עם facets/מיון/כותרת ישנים).
///
/// ה-extra_facets ממוינים ומנוקי-כפילויות — סדרם אינו משפיע על האינדקס.
/// Never returns 0. Pure string computation — safe to call synchronously.
#[frb(sync)]
pub fn compute_book_fingerprint(
    text: String,
    title: String,
    topics: String,
    catalogue_order: u32,
    generation_order: u32,
    extra_facets: Option<Vec<String>>,
) -> u64 {
    book_fingerprint(
        &text,
        &title,
        &topics,
        catalogue_order,
        generation_order,
        extra_facets.unwrap_or_default(),
    )
}

/// Borrowing form of [`compute_book_fingerprint`] so the indexing path
/// never clones a whole book to hash it.
fn book_fingerprint(
    text: &str,
    title: &str,
    topics: &str,
    catalogue_order: u32,
    generation_order: u32,
    mut extra_facets: Vec<String>,
) -> u64 {
    extra_facets.sort_unstable();
    extra_facets.dedup();
    let mut fnv = Fnv::new();
    fnv.feed_field(text.as_bytes());
    fnv.feed_field(title.as_bytes());
    fnv.feed_field(topics.as_bytes());
    fnv.feed_field(&catalogue_order.to_le_bytes());
    fnv.feed_field(&generation_order.to_le_bytes());
    for facet in &extra_facets {
        fnv.feed_field(facet.as_bytes());
    }
    fnv.finish()
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a מצטבר — הבסיס לכל חתימות התוכן. `feed_field` מקדים קידומת-אורך
/// כדי ששרשורי שדות שונים לא יתלכדו ("אב"+"ג" מול "א"+"בג").
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(FNV_OFFSET)
    }

    fn feed(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    fn feed_field(&mut self, bytes: &[u8]) {
        self.feed(&(bytes.len() as u64).to_le_bytes());
        self.feed(bytes);
    }

    /// 0 שמור ל"אין חתימה" — לעולם אינו מוחזר כחתימה אמיתית.
    fn finish(self) -> u64 {
        if self.0 == 0 {
            1
        } else {
            self.0
        }
    }
}

/// שורש-ממד שמור בנתיבי facet: נתיב שהמקטע הראשון שלו הוא אחד מאלה שייך
/// לממד סינון (מחבר/תקופה/ספרי-יסוד) ולא לעץ הקטגוריות. הסמנטיקה בסינון
/// (ראו [`SearchEngine::facet_filter_query`]): נתיבים מאותו ממד הם OR
/// ביניהם, וממדים שונים (וכן קבוצת הקטגוריות) הם AND — "ראשונים AND
/// מסכת ברכות" עובד, בעוד ריבוי מחברים נשאר "אחד מהם".
///
/// השורשים באנגלית בכוונה: שמות קטגוריות בספרייה הם עבריים, כך שאין
/// התנגשות עם עץ הקטגוריות הקיים.
pub const FACET_DIMENSION_ROOTS: [&str; 3] = ["author", "era", "base"];

/// תקרת חברות-הקבוצה המוחזרות לכל תוצאה מאוחדת ([`SearchResult::merged`]).
/// `merged_count` מדווח את הגודל האמיתי גם כשהרשימה נחתכת — למעט תחת
/// `truncated`, שם הוא תחתית (ראו [`SearchResult::merged_count`]).
const MERGED_SIBLINGS_CAP: usize = 10;

/// תקרת הקבוצות שהצבירה **הגלובלית** של [`GroupCollector`] מחזיקה (מפה
/// משותפת אחת לכל החיפוש, לא פר-סגמנט). הזיכרון של הקיבוץ פרופורציונלי
/// למספר הקבוצות הייחודיות — בלי תקרה, חיפוש exact של מילה נפוצה
/// (TermQuery, שאינו עובר דרך תקציב ה-postings של מילה-בודדת) צובר
/// GroupAcc לכל שורה ייחודית: ~100 בייט לקבוצה, מאות MB בשאילתות של
/// מיליוני שורות. בהגעה לתקרה — degrade, לא שגיאה: הקבוצה *הגרועה* לפי
/// סדר המיון מפנה את מקומה לקבוצה טובה ממנה (ראו [`BoundedGroups`]) — כך
/// שכל עמוד בטווח התקרה נשאר מדויק — ו-`truncated` מדווח למעלה (אותו דגל
/// "תוצאות חלקיות" שה-UI כבר מציג). `raw_total` נשאר מדויק. הזיכרון:
/// מפה גלובלית חסומת-50k (מפתח + GroupAcc עם עד [`MERGED_SIBLINGS_CAP`]
/// siblings + אינדקס BTreeSet) בתוספת buffers חסומים פר-סגמנט
/// ([`GROUP_FLUSH_BUFFER`]) — חסום, ללא תלות במספר הסגמנטים.
const GROUP_COLLECTOR_MAX_GROUPS: usize = 50_000;

/// תקציב הסעיפים הנספרים במסלול ההתאמה החלקית של טווח "תחת אותה כותרת" —
/// מילה נפוצה חותכת מאות אלפי סעיפים והמפה המשותפת גדלה כאיחוד שלהן.
/// מעבר לתקציב סעיפים *חדשים* נשמטים (degrade מדווח ב-`truncated`, כמו
/// [`GROUP_COLLECTOR_MAX_GROUPS`]) וסעיפים שכבר נספרים ממשיכים להצטבר —
/// המפה חסומה ל-500k רשומות `(u64, usize)`.
const SECTION_COUNT_BUDGET: usize = 500_000;

/// צובר את `sections` לתוך מפת הספירה תחת התקציב; מחזיר האם נחתך.
fn accumulate_section_counts(
    counts: &mut HashMap<u64, usize>,
    sections: HashSet<u64>,
    budget: usize,
) -> bool {
    let mut truncated = false;
    for section in sections {
        if let Some(count) = counts.get_mut(&section) {
            *count += 1;
        } else if counts.len() < budget {
            counts.insert(section, 1);
        } else {
            truncated = true;
        }
    }
    truncated
}

/// מינימום אותיות עבריות לחתימת דה-דופליקציה: שורה קצרה מזה מקבלת 0
/// ("אין חתימה") ולעולם לא תתאחד עם שורות אחרות במצב `IdenticalText` —
/// כותרות ושורות בנות מילה-שתיים זהות בכל מקום ואיחודן היה מטעה.
const LINE_DEDUP_MIN_LETTERS: usize = 12;

/// חתימת דה-דופליקציה לשורת אינדקס: FNV-1a על האותיות והספרות (כל
/// יוניקוד, מקופלות-רישיות) של הטקסט המנורמל — רווחים, פיסוק, גרשיים
/// וסימנים אחרים אינם משתתפים, כך ששתי מהדורות שנבדלות רק
/// בפיסוק/רווחים מקבלות אותה חתימה. ספרות ואותיות זרות *כן* משתתפות:
/// "לשלם 100 שקלים" ו"לשלם 200 שקלים" הן שורות שונות, לא כפילות (המחיר:
/// מהדורה שמוסיפה מספרי-פסוק בספרות לא תתאחד עם מהדורה נקייה — עדיף
/// פיצול-יתר על איחוד שמעלים תוכן). שינויי כתיב (מלא/חסר) גם הם משנים
/// את החתימה — זהו דה-דופ שמרני של טקסט זהה, לא זיהוי מקבילות מקורב.
/// מוטבעת בשדה `lineHash` (FAST) בזמן אינדוקס ומשמשת את מצב האיחוד
/// [`ResultGrouping::IdenticalText`].
///
/// 0 שמור ל"אין חתימה" — פחות מ-[`LINE_DEDUP_MIN_LETTERS`] אותיות
/// *עבריות* (אלפאנומרי משתתף בחתימה אך אינו נספר לסף).
fn line_dedup_hash(normalized_text: &str) -> u64 {
    let mut fnv = Fnv::new();
    let mut letters = 0usize;
    let mut buf = [0u8; 4];
    for c in normalized_text.chars() {
        if ('א'..='ת').contains(&c) {
            letters += 1;
        } else if !c.is_alphanumeric() {
            continue;
        }
        for lower in c.to_lowercase() {
            fnv.feed(lower.encode_utf8(&mut buf).as_bytes());
        }
    }
    if letters < LINE_DEDUP_MIN_LETTERS {
        return 0;
    }
    fnv.finish()
}

/// Mirrors the Dart `IndexingDocumentBuilder._updateReferenceTrail`: a new
/// `<h…>` heading line replaces any earlier trail entry that shares its
/// first four characters (same heading level) and everything after it, then
/// appends itself. Char-based like the Dart UTF-16 indexing (identical for
/// BMP text, which Hebrew books are).
fn update_reference_trail<'a>(trail: &mut Vec<&'a str>, line: &'a str) {
    if line.chars().count() >= 4 && !trail.is_empty() {
        let prefix: Vec<char> = line.chars().take(4).collect();
        if let Some(idx) = trail.iter().position(|entry| {
            entry.chars().take(4).eq(prefix.iter().copied()) && entry.chars().count() >= 4
        }) {
            trail.truncate(idx);
        }
    }
    trail.push(line);
}

fn check_index_compatibility_path(index_path: &Path) -> IndexCompatibility {
    let metadata_path = index_metadata_path(index_path);

    if !index_path.exists() {
        return compatibility(
            false,
            "missing_index",
            None,
            metadata_path,
            Some("index directory does not exist".to_string()),
        );
    }

    if !index_path.is_dir() {
        return compatibility(
            false,
            "invalid_index_path",
            None,
            metadata_path,
            Some("index path is not a directory".to_string()),
        );
    }

    if metadata_path.exists() {
        return check_sidecar_metadata(index_path, metadata_path);
    }

    check_legacy_tantivy_metadata(index_path, metadata_path)
}

/// `Some(reason)` כשלא ניתן לאשר שהסכימה השמורה ב-meta.json של tantivy זהה
/// לסכימת המנוע הנוכחית — בדיוק ההשוואה ש-`Index::open_or_create` מבצע.
/// בלעדיה, קובץ צדדי שמצהיר על הגרסה הנכונה עובר את בדיקת התאימות בעוד
/// שפתיחת המנוע עדיין נופלת על SchemaError (אינדקס שנבנה בגרסת-ביניים של
/// אותה schema_version) — והאפליקציה נופלת בשקט לאינדקס זמני.
/// גם meta.json חסר/פגום נחשב אי-התאמה: sidecar תקין לא מעיד כלום כשה-metadata
/// של tantivy עצמו לא קריא, ופתיחת האינדקס תיכשל באותה מידה.
fn stored_schema_mismatch(index_path: &Path) -> Option<String> {
    let raw = match fs::read_to_string(index_path.join("meta.json")) {
        Ok(raw) => raw,
        Err(err) => return Some(format!("tantivy meta.json is missing or unreadable: {err}")),
    };
    let meta: JsonValue = match serde_json::from_str(&raw) {
        Ok(meta) => meta,
        Err(err) => return Some(format!("tantivy meta.json is not valid JSON: {err}")),
    };
    let Some(schema_json) = meta.get("schema").cloned() else {
        return Some("tantivy meta.json has no schema entry".to_string());
    };
    match serde_json::from_value::<Schema>(schema_json) {
        Ok(stored) if stored == current_schema() => None,
        Ok(_) => Some("tantivy schema on disk differs from the engine schema".to_string()),
        Err(err) => Some(format!("stored tantivy schema is unreadable: {err}")),
    }
}

fn check_sidecar_metadata(index_path: &Path, metadata_path: PathBuf) -> IndexCompatibility {
    let raw = match fs::read_to_string(&metadata_path) {
        Ok(raw) => raw,
        Err(err) => {
            return compatibility(
                false,
                "invalid_metadata",
                None,
                metadata_path,
                Some(format!("failed to read metadata: {err}")),
            )
        }
    };

    let metadata: IndexMetadata = match serde_json::from_str(&raw) {
        Ok(metadata) => metadata,
        Err(err) => {
            return compatibility(
                false,
                "invalid_metadata",
                None,
                metadata_path,
                Some(format!("failed to parse metadata: {err}")),
            )
        }
    };

    if metadata.format != INDEX_FORMAT {
        return compatibility(
            false,
            "invalid_format",
            Some(metadata.schema_version),
            metadata_path,
            Some(format!("expected format {INDEX_FORMAT}")),
        );
    }

    if metadata.schema_version < INDEX_SCHEMA_VERSION {
        return compatibility(
            false,
            "rebuild_required",
            Some(metadata.schema_version),
            metadata_path,
            Some("index schema is older than the engine requires".to_string()),
        );
    }

    if metadata.schema_version > INDEX_SCHEMA_VERSION {
        return compatibility(
            false,
            "engine_too_old",
            Some(metadata.schema_version),
            metadata_path,
            Some("index schema is newer than this engine supports".to_string()),
        );
    }

    if let Some(reason) = stored_schema_mismatch(index_path) {
        return compatibility(
            false,
            "rebuild_required",
            Some(metadata.schema_version),
            metadata_path,
            Some(reason),
        );
    }

    compatibility(
        true,
        "compatible",
        Some(metadata.schema_version),
        metadata_path,
        None,
    )
}

fn check_legacy_tantivy_metadata(index_path: &Path, metadata_path: PathBuf) -> IndexCompatibility {
    let tantivy_metadata_path = index_path.join("meta.json");
    if !tantivy_metadata_path.exists() {
        return compatibility(
            false,
            "missing_metadata",
            None,
            metadata_path,
            Some("otzaria metadata and Tantivy meta.json are missing".to_string()),
        );
    }

    let raw = match fs::read_to_string(&tantivy_metadata_path) {
        Ok(raw) => raw,
        Err(err) => {
            return compatibility(
                false,
                "invalid_tantivy_metadata",
                None,
                metadata_path,
                Some(format!("failed to read Tantivy metadata: {err}")),
            )
        }
    };

    let tantivy_metadata: JsonValue = match serde_json::from_str(&raw) {
        Ok(metadata) => metadata,
        Err(err) => {
            return compatibility(
                false,
                "invalid_tantivy_metadata",
                None,
                metadata_path,
                Some(format!("failed to parse Tantivy metadata: {err}")),
            )
        }
    };

    if tantivy_schema_matches_current_version(&tantivy_metadata) {
        return compatibility(
            true,
            "legacy_compatible",
            Some(INDEX_SCHEMA_VERSION),
            metadata_path,
            Some(
                "otzaria metadata is missing, but Tantivy schema matches the current engine"
                    .to_string(),
            ),
        );
    }

    compatibility(
        false,
        "rebuild_required",
        inferred_legacy_schema_version(&tantivy_metadata),
        metadata_path,
        Some("otzaria metadata is missing and Tantivy schema is not compatible".to_string()),
    )
}

/// Compares the full on-disk schema against the engine's current one — the
/// same equality `Index::open_or_create` enforces — so a legacy index can't
/// pass the check (e.g. on the `id` field alone) and then fail to open.
fn tantivy_schema_matches_current_version(metadata: &JsonValue) -> bool {
    let Some(schema_json) = metadata.get("schema") else {
        return false;
    };
    match serde_json::from_value::<Schema>(schema_json.clone()) {
        Ok(found_schema) => found_schema == current_schema(),
        Err(_) => false,
    }
}

fn inferred_legacy_schema_version(metadata: &JsonValue) -> Option<u32> {
    let schema = metadata.get("schema")?.as_array()?;
    let id_field = schema.iter().find(|field| {
        field.get("name").and_then(JsonValue::as_str) == Some("id")
            && field.get("type").and_then(JsonValue::as_str) == Some("u64")
    })?;
    if id_field
        .pointer("/options/indexed")
        .and_then(JsonValue::as_bool)
        == Some(false)
    {
        Some(1)
    } else {
        None
    }
}

fn ensure_current_index_metadata(index_path: &Path) -> Result<()> {
    let compatibility = check_index_compatibility_path(index_path);
    if compatibility.compatible && compatibility.status != "compatible" {
        write_current_index_metadata(index_path)?;
    }
    Ok(())
}

fn write_current_index_metadata(index_path: &Path) -> Result<()> {
    let metadata_path = index_metadata_path(index_path);
    let serialized = serde_json::to_string_pretty(&current_index_metadata())?;
    fs::write(&metadata_path, format!("{serialized}\n")).with_context(|| {
        format!(
            "failed to write index metadata to {}",
            metadata_path.display()
        )
    })
}

fn current_index_metadata() -> IndexMetadata {
    IndexMetadata {
        format: INDEX_FORMAT.to_string(),
        schema_version: INDEX_SCHEMA_VERSION,
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        tantivy_version: TANTIVY_INDEX_VERSION.to_string(),
        created_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    }
}

fn index_metadata_path(index_path: &Path) -> PathBuf {
    index_path.join(INDEX_METADATA_FILE_NAME)
}

fn compatibility(
    compatible: bool,
    status: &str,
    found_schema_version: Option<u32>,
    metadata_path: PathBuf,
    reason: Option<String>,
) -> IndexCompatibility {
    IndexCompatibility {
        compatible,
        status: status.to_string(),
        found_schema_version,
        required_schema_version: INDEX_SCHEMA_VERSION,
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        metadata_path: metadata_path.display().to_string(),
        reason,
    }
}

/// The schema this engine version requires. Kept in one place so `new()` and
/// the legacy compatibility check can never drift apart.
fn current_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    // Deliberately NOT fast: a text fast field stores every raw line in a
    // columnar dictionary — a second full copy of the corpus — and nothing
    // reads it (collectors use only the filePath/contentHash/id columns).
    schema_builder.add_text_field(
        "text",
        TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("hebrew")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored(),
    );
    // השדה המנוקד: מאוכלס רק בשורות שנושאות ניקוד/טעמים (שאר השורות פשוט
    // אינן קיימות בו — tantivy מטפל בשדה חסר בחינם). מאונדקס בטוקנייזר
    // ששומר את הסימנים, ומאוחסן כדי שתוצאות חיפוש מנוקד יציגו את הטקסט
    // המנוקד. חיפוש רגיל אינו נוגע בשדה הזה כלל.
    schema_builder.add_text_field(
        "textVocalized",
        TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("hebrew_vocalized")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored(),
    );
    schema_builder.add_text_field("reference", STORED);
    schema_builder.add_text_field(
        "title",
        TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_fieldnorms(false),
            )
            .set_stored(),
    );
    // INDEXED is required for delete_term / upsert by id to work.
    schema_builder.add_u64_field("id", STORED | FAST | INDEXED);
    schema_builder.add_u64_field("segment", STORED);
    schema_builder.add_bool_field("isPdf", STORED);
    schema_builder.add_text_field("filePath", STRING | FAST | STORED);
    // Book-level content fingerprint, stamped identically on all documents of
    // a book. FAST-only: read columnar by get_book_fingerprints, never searched
    // or returned. 0 = no fingerprint recorded.
    schema_builder.add_u64_field("contentHash", FAST);
    // חתימת טקסט-בלבד של הספר (content_fingerprint על הטקסט הגולמי), זהה על
    // כל מסמכי הספר. FAST בלבד — נקראת עמודתית ע"י get_book_text_fingerprints.
    schema_builder.add_u64_field("textHash", FAST);
    // מזהה סעיף: כל השורות שתחת אותה כותרת (אותו בלוק reference) נושאות
    // אותו ערך, ייחודי גלובלית — id_base של הספר + אינדקס בלוק הכותרת
    // (ב-PDF: מספר העמוד). FAST בלבד: מסלול "תחת אותה כותרת"
    // (SearchScope::SameSection) קורא אותו עמודתית לחיתוך סעיפים; אינו
    // מאוחסן ואינו מחופש ישירות.
    schema_builder.add_u64_field("sectionId", FAST);
    schema_builder.add_u64_field("generationSort", FAST);
    // חתימת דה-דופליקציה של גוף השורה (ראו line_dedup_hash). FAST בלבד:
    // נקראת עמודתית ע"י קיבוץ IdenticalText; אינה מאוחסנת ואינה מחופשת.
    schema_builder.add_u64_field("lineHash", FAST);
    schema_builder.add_facet_field("topics", FacetOptions::default());
    schema_builder.build()
}

/// Cache key for materialized single-word term sets. The searcher
/// `generation_id` changes on every reader reload (i.e. after each commit),
/// so entries from a stale index snapshot can never be served.
#[derive(Hash, PartialEq, Eq)]
struct TermCacheKey {
    generation: u64,
    /// The dictionary the branches were scanned against (`text` vs
    /// `textVocalized`) — identical branch strings on different fields
    /// materialize different term sets.
    field: Field,
    branches: Vec<String>,
    /// Tokens expanded via Levenshtein-1 automaton scans (single-word typo
    /// path); part of the key because the same branches with different typo
    /// tokens materialize different term sets.
    typo_tokens: Vec<String>,
    max_expansions: u32,
    /// The postings-cost ceiling the terms were collected under (differs
    /// between the single-word path and the phrase-position path, which can
    /// otherwise share every other key component through the string API).
    postings_budget: u64,
}

/// Entries kept in [`SearchEngine::term_cache`]. Each entry holds at most
/// `max_expansions` terms (≤50 000 short Hebrew tokens ≈ ~1MB worst case), so
/// the cache tops out at a few tens of MB when 32 distinct worst-case queries
/// are live — typically far less (the postings budget truncates collection
/// long before the term ceiling on any realistic query).
const TERM_CACHE_ENTRIES: usize = 32;

/// A materialized single-word term set plus whether collection stopped early
/// on a budget overflow. Cached together so every engine call behind one user
/// search (stream + counts + count-by-book + facets + pagination) reports the
/// same truncation state without re-scanning the FST dictionary.
#[derive(Clone)]
struct CachedTermSet {
    terms: Arc<Vec<Term>>,
    truncated: bool,
}

/// Ceiling on the summed per-segment `doc_freq` a single-word term set may
/// accumulate. This — not the term count — is the true cost guard: executing
/// the resulting `TermSetQuery` unions one postings list per matched term per
/// segment into a BitSet, O(Σ doc_freq). The term-count ceiling
/// (`max_expansions`) remains only as a memory guard on the materialized
/// `Vec<Term>`. Initial value pending empirical calibration via
/// `benchmark_cli` (VARIATION_CEILING_RESEARCH.md §3.א).
const SINGLE_WORD_POSTINGS_BUDGET: u64 = 1_000_000;

/// Term ceiling for one *word position* of a phrase on the degrade path
/// ([`TermListPhraseQuery`]). Unlike tantivy's `RegexPhraseQuery` ceiling —
/// which is cumulative across every position and *errors* on overflow — this
/// bounds each position independently and overflow degrades: collection
/// truncates from the back of the branch-priority order, exactly like the
/// single-word path. The value bounds the materialized `Vec<Term>` and the
/// per-segment positional postings the verifier opens (one per matched term
/// per position), so it is a memory guard, not a scan-time guard.
const PHRASE_POSITION_MAX_EXPANSIONS: u32 = 8_192;

/// Postings-cost ceiling for one phrase position on the degrade path.
/// Deliberately far looser than [`SINGLE_WORD_POSTINGS_BUDGET`]: the exact
/// `RegexPhraseQuery` path this replaces never bounded Σ doc_freq at all (its
/// bucketed unions contain the post-expansion cost), so a tight budget here
/// would truncate phrases that used to work. This exists only as a backstop
/// against a pathological position (e.g. a 1-char word with prefix+suffix
/// windows) unioning a huge slice of the index.
const PHRASE_POSITION_POSTINGS_BUDGET: u64 = 20_000_000;

// ── Vocalized-path expansion ceilings ──────────────────────────────────────
// Vocalized patterns are always expansions (free-mark runs match every
// vocalization of the word), so even "exact" vocalized search materializes a
// term set. The postings budget above remains the true cost guard; these
// only bound the materialized `Vec<Term>` / phrase-expansion memory.

/// Term ceiling for one exact vocalized word (`TermSetQuery` path).
const VOC_EXACT_SINGLE_MAX_EXPANSIONS: u32 = 4_096;
/// Cumulative expansion ceiling for a vocalized phrase (`RegexPhraseQuery`).
const VOC_PHRASE_MAX_EXPANSIONS: u32 = 8_192;
/// Term ceiling for one fuzzy vocalized word (exact + lexical + edit-distance
/// branches share it; overflow degrades like the advanced single-word path).
const VOC_FUZZY_MAX_EXPANSIONS: u32 = 20_000;
/// Cap on plain-dictionary variants collected per token by the vocalized
/// fuzzy/typo expansion (the Levenshtein scan runs on mark-free bases).
const VOC_VARIANTS_PER_TOKEN: usize = 128;

/// Mirrors the pinned sidecar's semantic candidate ceiling. Applying the same
/// bound before constructing Tantivy's `TopDocs` collector prevents hostile or
/// accidental `limit`/`offset` values from driving an unbounded allocation.
#[cfg(feature = "semantic-integration")]
const MAX_SEMANTIC_CANDIDATE_WINDOW: u32 = 10_000;

pub struct SearchEngine {
    schema: Schema,
    /// The directory this engine opened. Retained only so a build can ask whether the
    /// index it is about to read is one this version reads — see
    /// [`Self::index_compatibility`].
    #[cfg_attr(not(feature = "semantic-integration"), allow(dead_code))]
    index_path: PathBuf,
    index: Index,
    index_writer: Option<IndexWriter>,
    writer_heap_size: usize,
    index_reader: IndexReader,
    /// Optional lexical morphology lexicon for the approximate (`fuzzy`) path.
    /// `None` until [`SearchEngine::set_magic_dictionary_path`] loads a valid
    /// `lexical.db`; while `None`, fuzzy search behaves exactly as before.
    magic_dict: Option<MagicDictionary>,
    /// מילון תרגום ארמי↔עברי לאפשרות "תרגום ארמי" של החיפוש המתקדם.
    /// `None` עד ש-[`SearchEngine::set_translation_dictionary_path`] טוען
    /// קובץ תקין; בהיעדרו האפשרות פשוט לא מרחיבה דבר.
    translation_dict: Option<TranslationLexicon>,
    /// מילון פענוח ראשי-תיבות לאפשרות "ראשי תיבות" של החיפוש המתקדם.
    /// `None` עד ש-[`SearchEngine::set_acronyms_dictionary_path`] טוען קובץ
    /// תקין; בהיעדרו האפשרות פשוט לא מרחיבה דבר.
    acronym_dict: Option<AcronymLexicon>,
    /// Materialized-terms cache for the single-word regex path. One user
    /// search triggers several engine calls with identical parameters
    /// (stream + count + count-by-book + facet counts + pagination), and the
    /// FST dictionary scan behind `single_regex_term_query` is the expensive
    /// part of each — this makes every call after the first near-free.
    term_cache: Mutex<LruCache<TermCacheKey, CachedTermSet>>,
    /// Bulk-indexing mode (see [`SearchEngine::set_bulk_indexing`]): while
    /// on, the live writer (and any lazily-reopened one) uses `NoMergePolicy`.
    bulk_indexing: bool,
    /// Optional semantic sidecar. It owns vector retrieval, fusion, grouping
    /// and semantic-index lifecycle; Tantivy remains owned by this engine.
    #[cfg_attr(not(feature = "semantic-integration"), allow(dead_code))]
    semantic_runtime: SemanticRuntime,
}

/// Installs a stderr logger (once per process) so the engine's `info!`
/// timing logs are visible in the app console without any Dart-side setup.
/// `RUST_LOG` still overrides the default filter; if a logger is already
/// installed (tests, benchmark_cli), `try_init` leaves it in place.
fn init_engine_logger() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or("search_engine=info"),
        )
        .try_init();
    });
}

impl SearchEngine {
    /// Deliberately **not** `#[frb(sync)]` — opening the index mmaps and reads every
    /// segment footer; a synchronous binding blocks the calling Dart isolate throughout.
    pub fn new(path: &str) -> Self {
        init_engine_logger();
        debug!("new path={}", path);
        let schema = current_schema();
        let mmap_directory = MmapDirectory::open(path).expect("unable to open mmap directory");
        let index = match Index::open_or_create(mmap_directory, schema.clone()) {
            Ok(index) => index,
            Err(tantivy::TantivyError::SchemaError(err)) => panic!(
                "index at {path} was built with an incompatible schema ({err}); \
                 call check_index_compatibility before opening and rebuild the index"
            ),
            Err(err) => panic!("Failed to open index at {path}: {err}"),
        };
        // אנליזטורי השדות (מצב אינדוקס): מילה עם גרש/גרשיים מוטמעת גם
        // בצורתה הנקייה באותה עמדה — חיפוש `רמבם` מוצא `רמב"ם`.
        // RemoveLongFilter על *כל* האנליזטורים — כולל צד השאילתה, אחרת
        // ספירת הטוקנים ב-PhraseQuery סוטה מהאינדקס: זבל base64 שזלג
        // לקורפוס (נמדדו ריצות של 4,618 תווים) לא נכנס למילון הטרמים.
        // 128 בייט ≈ 64 אותיות עבריות — פי כמה מכל מילה לגיטימית.
        index.tokenizers().register(
            "hebrew",
            TextAnalyzer::builder(HebrewTokenizer {
                emit_quote_free: true,
                keep_marks: false,
            })
            .filter(RemoveLongFilter::limit(MAX_TOKEN_BYTES))
            .filter(LowerCaser)
            .build(),
        );
        index.tokenizers().register(
            "hebrew_vocalized",
            TextAnalyzer::builder(HebrewTokenizer {
                emit_quote_free: true,
                keep_marks: true,
            })
            .filter(RemoveLongFilter::limit(MAX_TOKEN_BYTES))
            .filter(LowerCaser)
            .build(),
        );
        // גרסאות צד-שאילתה: בלי הפליטה הכפולה — שאילתה מטוקננת לטוקן אחד
        // לכל מילה (מסלול ה-exact בונה PhraseQuery לפי מספר הטוקנים).
        index.tokenizers().register(
            "hebrew_query",
            TextAnalyzer::builder(HebrewTokenizer {
                emit_quote_free: false,
                keep_marks: false,
            })
            .filter(RemoveLongFilter::limit(MAX_TOKEN_BYTES))
            .filter(LowerCaser)
            .build(),
        );
        index.tokenizers().register(
            "hebrew_vocalized_query",
            TextAnalyzer::builder(HebrewTokenizer {
                emit_quote_free: false,
                keep_marks: true,
            })
            .filter(RemoveLongFilter::limit(MAX_TOKEN_BYTES))
            .filter(LowerCaser)
            .build(),
        );
        let index_reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .expect("Failed to create index reader");
        // Best-effort: if another instance/process holds the writer lock right
        // now, start without a writer; ensure_writer() retries on first write.
        let index_writer = match index.writer(DEFAULT_WRITER_HEAP_SIZE) {
            Ok(writer) => Some(writer),
            Err(err) => {
                warn!("writer unavailable at startup ({err}); will retry lazily");
                None
            }
        };

        if let Err(err) = ensure_current_index_metadata(Path::new(path)) {
            debug!("failed to ensure index metadata: {err:#}");
        }

        SearchEngine {
            schema,
            index_path: PathBuf::from(path),
            index,
            index_writer,
            writer_heap_size: DEFAULT_WRITER_HEAP_SIZE,
            index_reader,
            magic_dict: None,
            translation_dict: None,
            acronym_dict: None,
            term_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(TERM_CACHE_ENTRIES).expect("cache size is non-zero"),
            )),
            bulk_indexing: false,
            #[cfg(feature = "semantic-integration")]
            semantic_runtime: None,
            #[cfg(not(feature = "semantic-integration"))]
            semantic_runtime: (),
        }
    }

    /// Loads a `lexical.db` morphology lexicon for the approximate (`fuzzy`)
    /// search path. Returns `true` if the file opened and has the expected
    /// schema, `false` if it is missing or unusable — in which case the engine
    /// keeps its existing fuzzy behaviour (no error is surfaced, so the app can
    /// call this unconditionally at startup). Does **not** affect exact or
    /// advanced search.
    #[frb(sync)]
    pub fn set_magic_dictionary_path(&mut self, path: String) -> bool {
        match MagicDictionary::open(Path::new(&path)) {
            Ok(dict) => {
                debug!("magic dictionary loaded from {path}");
                self.magic_dict = Some(dict);
                true
            }
            Err(err) => {
                warn!("magic dictionary unavailable at {path}: {err:#}");
                self.magic_dict = None;
                false
            }
        }
    }

    /// Whether a lexical dictionary is currently loaded (i.e. approximate search
    /// will use morphological expansion).
    #[frb(sync)]
    pub fn has_magic_dictionary(&self) -> bool {
        self.magic_dict.is_some()
    }

    /// טוען את מילון התרגום הארמי-עברי (ה-`dictionary.json` של האפליקציה)
    /// עבור אפשרות "תרגום ארמי" בחיפוש המתקדם. מחזיר `true` אם הקובץ נטען;
    /// `false` אם חסר/פגום — ואז האפשרות לא מרחיבה דבר (אין שגיאה כלפי
    /// האפליקציה, שיכולה לקרוא לזה ללא תנאי באתחול).
    #[frb(sync)]
    pub fn set_translation_dictionary_path(&mut self, path: String) -> bool {
        match TranslationLexicon::load(Path::new(&path)) {
            Ok(lexicon) => {
                debug!(
                    "translation dictionary loaded from {path} ({} headwords)",
                    lexicon.len()
                );
                self.translation_dict = Some(lexicon);
                true
            }
            Err(err) => {
                warn!("translation dictionary unavailable at {path}: {err:#}");
                self.translation_dict = None;
                false
            }
        }
    }

    /// האם מילון תרגום ארמי-עברי טעון כרגע.
    #[frb(sync)]
    pub fn has_translation_dictionary(&self) -> bool {
        self.translation_dict.is_some()
    }

    /// טוען את מילון ראשי-התיבות (ה-`Acronyms.json` של האפליקציה) עבור
    /// אפשרות "ראשי תיבות" בחיפוש המתקדם. מחזיר `true` אם הקובץ נטען;
    /// `false` אם חסר/פגום — ואז האפשרות לא מרחיבה דבר (אין שגיאה כלפי
    /// האפליקציה, שיכולה לקרוא לזה ללא תנאי באתחול).
    #[frb(sync)]
    pub fn set_acronyms_dictionary_path(&mut self, path: String) -> bool {
        match AcronymLexicon::load(Path::new(&path)) {
            Ok(lexicon) => {
                debug!(
                    "acronyms dictionary loaded from {path} ({} acronyms)",
                    lexicon.len()
                );
                self.acronym_dict = Some(lexicon);
                true
            }
            Err(err) => {
                warn!("acronyms dictionary unavailable at {path}: {err:#}");
                self.acronym_dict = None;
                false
            }
        }
    }

    /// האם מילון ראשי-תיבות טעון כרגע.
    #[frb(sync)]
    pub fn has_acronyms_dictionary(&self) -> bool {
        self.acronym_dict.is_some()
    }

    // ── Semantic sidecar API ────────────────────────────────────────────────

    /// Open the semantic sidecar and wire it to the already-open Tantivy
    /// engine. The sidecar owns semantic fusion; Tantivy stays owned here. When
    /// this crate was built without the optional semantic feature this is a
    /// no-op that returns an explicit Disabled status.
    ///
    /// **The sidecar's vector store is in-memory** (check
    /// [`SemanticStatus::vectors_persisted`]): vectors live only for the
    /// lifetime of this session and must be rebuilt after a restart. Opening an
    /// engine re-reads the on-disk manifest and drops every book record whose
    /// vectors did not survive, so a *re-open* discards the session's semantic
    /// index. This method therefore does not re-open:
    ///
    /// - Called again with the same inputs it is a no-op returning the current
    ///   status, so a caller that configures defensively cannot lose an index.
    /// - Called with different inputs while a session is open it fails and says
    ///   which input changed. Switching model or library root is an explicit
    ///   act: call [`Self::disable_semantic`] first and accept the rebuild.
    pub fn configure_semantic(&mut self, config: SemanticConfigInput) -> Result<SemanticStatus> {
        #[cfg(feature = "semantic-integration")]
        {
            let requested = SemanticConfigKey::from_input(&config);
            if let Some(active) = &self.semantic_runtime {
                if active.config == requested {
                    return Ok(self.semantic_status());
                }
                return Err(anyhow::anyhow!(
                    "the semantic sidecar is already configured and {} changed; the vector \
                     store is in-memory, so re-opening would discard the vectors indexed in \
                     this session. Call disable_semantic() first if that is intended",
                    active.config.changed_fields(&requested)
                ));
            }

            let mut semantic_config = SemanticConfig {
                root_dir: requested.root_dir.clone(),
                model_path: requested.model_path.clone(),
                embedding_model_id: requested.model_id.clone(),
                embedding_dim: requested.embedding_dim,
                ..SemanticConfig::default()
            };
            // Both dimensions must agree or the sidecar refuses the config, and
            // the store's path is derived from our root rather than the
            // sidecar's own default root.
            semantic_config.store.embedding_dim = requested.embedding_dim;
            semantic_config.store.db_path = requested.root_dir.join("vectors");

            let engine = SemanticEngine::open(semantic_config)
                .map_err(|err| anyhow::anyhow!("failed to open semantic sidecar: {err}"))?;
            self.semantic_runtime = Some(ConfiguredSemantic {
                engine: OtzariaHybridEngine::new(HybridCoordinator::new(Some(engine))),
                config: requested,
            });
            Ok(self.semantic_status())
        }

        #[cfg(not(feature = "semantic-integration"))]
        {
            let _ = config;
            Ok(Self::semantic_disabled_status())
        }
    }

    /// Remove the configured sidecar without touching its on-disk files.
    /// This is useful when an app switches library roots or wants lexical-only
    /// operation for the current session, and it is the explicit way to allow a
    /// subsequent [`Self::configure_semantic`] with different inputs.
    ///
    /// Because the vector store is in-memory, this drops the session's vectors:
    /// re-configuring afterwards needs a full semantic re-index.
    pub fn disable_semantic(&mut self) {
        #[cfg(feature = "semantic-integration")]
        {
            self.semantic_runtime = None;
        }
    }

    /// The open sidecar engine, or `None` when semantic search has not been
    /// configured in this session.
    #[cfg(feature = "semantic-integration")]
    fn semantic_engine(&self) -> Option<&OtzariaHybridEngine> {
        self.semantic_runtime.as_ref().map(|active| &active.engine)
    }

    /// Deliberately **not** `#[frb(sync)]`. Reading the status takes the
    /// sidecar's engine lock, which indexing holds for the duration of one
    /// book's embedding run; a synchronous binding would block the calling Dart
    /// isolate for that whole time. Progress polling is the expected caller, so
    /// it must not be able to freeze the UI.
    pub fn semantic_status(&self) -> SemanticStatus {
        #[cfg(feature = "semantic-integration")]
        {
            if let Some(runtime) = self.semantic_engine() {
                let status = runtime.get_semantic_status();
                return SemanticStatus {
                    enabled: true,
                    available: status.available,
                    model_loaded: status.model_loaded,
                    indexed_book_count: status.indexed_book_count,
                    vector_count: status.vector_count,
                    model_id: status.model_id,
                    embedding_dim: status.embedding_dim,
                    embedding_backend: status.embedding_backend,
                    vector_backend: status.vector_backend,
                    vectors_persisted: status.vectors_persisted,
                    needs_full_reindex: status.needs_full_reindex,
                    last_error: status.last_error,
                };
            }
            Self::semantic_not_configured_status()
        }

        #[cfg(not(feature = "semantic-integration"))]
        {
            Self::semantic_disabled_status()
        }
    }

    /// Index or replace semantic vectors for complete books. The caller should
    /// use the same fingerprint it uses in `semantic_index_diff`; line ids must
    /// be the global Tantivy document ids so semantic-only results can hydrate.
    ///
    /// Takes `&self` on purpose. It mutates only the sidecar, which serializes
    /// indexing behind its own mutex and releases the engine lock between
    /// books. Declaring `&mut self` would make flutter_rust_bridge take a write
    /// lock on the whole engine for the entire run, blocking every concurrent
    /// *lexical* search for as long as the library takes to embed.
    pub fn semantic_index_books(
        &self,
        books: Vec<SemanticBookInput>,
    ) -> Result<SemanticIndexingSummary> {
        #[cfg(feature = "semantic-integration")]
        {
            let Some(runtime) = self.semantic_engine() else {
                return Ok(SemanticIndexingSummary {
                    enabled: false,
                    books_indexed: 0,
                    books_skipped: 0,
                    books_empty: 0,
                    chunks_written: 0,
                });
            };
            let sidecar_books: Vec<SidecarBookForIndexing> = books
                .into_iter()
                .map(|book| SidecarBookForIndexing {
                    source_book_key: book.source_book_key,
                    title: book.title,
                    content_fingerprint: book.content_fingerprint,
                    is_pdf: book.is_pdf,
                    topics: book.topics,
                    extra_facets: book.extra_facets,
                    lines: book
                        .lines
                        .into_iter()
                        .map(|line| SidecarBookLine {
                            line_id: line.line_id,
                            section_id: line.section_id,
                            text: line.text,
                            line_hash: line.line_hash,
                            reference: line.reference,
                            segment: line.segment,
                        })
                        .collect(),
                })
                .collect();
            let summary = runtime
                .index_books(&sidecar_books)
                .map_err(|err| anyhow::anyhow!("semantic indexing failed: {err}"))?;
            let summary = summary.unwrap_or_default();
            Ok(SemanticIndexingSummary {
                enabled: true,
                books_indexed: summary.books_indexed,
                books_skipped: summary.books_skipped,
                books_empty: summary.books_empty,
                chunks_written: summary.chunks_written,
            })
        }

        #[cfg(not(feature = "semantic-integration"))]
        {
            let _ = books;
            Ok(SemanticIndexingSummary {
                enabled: false,
                books_indexed: 0,
                books_skipped: 0,
                books_empty: 0,
                chunks_written: 0,
            })
        }
    }

    /// Compare the semantic manifest with the book fingerprints stored in the
    /// lexical index. A `contentHash` of zero is deliberately surfaced as
    /// `unverifiable_books` rather than treated as an up-to-date PDF.
    pub fn semantic_index_diff(&self) -> Result<SemanticIndexDiff> {
        #[cfg(feature = "semantic-integration")]
        {
            let Some(runtime) = self.semantic_engine() else {
                return Ok(Self::semantic_disabled_diff());
            };
            let fingerprints = self.get_book_fingerprints()?;
            // The sidecar now separates "the semantic path is off" from "the comparison
            // itself failed": the first is `Ok(None)`, the second an error. Collapsing
            // them would report a broken manifest as a disabled feature, and the app
            // would offer indexing as the fix.
            let diff = runtime
                .get_semantic_index_diff_from_lexical_hashes(&fingerprints)
                .map_err(|err| anyhow::anyhow!("semantic index diff failed: {err}"))?;
            let Some(diff) = diff else {
                return Ok(Self::semantic_disabled_diff());
            };
            Ok(SemanticIndexDiff {
                enabled: true,
                new_books: diff.new_books,
                changed_books: diff.changed_books,
                unverifiable_books: diff.unverifiable_books,
                removed_books: diff.removed_books,
                model_mismatch: diff.model_mismatch,
                chunking_mismatch: diff.chunking_mismatch,
                normalization_mismatch: diff.normalization_mismatch,
            })
        }

        #[cfg(not(feature = "semantic-integration"))]
        {
            Ok(Self::semantic_disabled_diff())
        }
    }

    /// Remove vector records for books previously reported as `removed_books`.
    /// This never deletes lexical Tantivy documents.
    ///
    /// `&self` for the same reason as [`Self::semantic_index_books`].
    pub fn remove_semantic_books(
        &self,
        source_book_keys: Vec<String>,
    ) -> Result<SemanticRemoveResult> {
        #[cfg(feature = "semantic-integration")]
        {
            let Some(runtime) = self.semantic_engine() else {
                return Ok(SemanticRemoveResult {
                    enabled: false,
                    vectors_removed: 0,
                });
            };
            let removed = runtime
                .remove_semantic_books(&source_book_keys)
                .map_err(|err| anyhow::anyhow!("semantic remove failed: {err}"))?
                .unwrap_or(0);
            Ok(SemanticRemoveResult {
                enabled: true,
                vectors_removed: removed,
            })
        }

        #[cfg(not(feature = "semantic-integration"))]
        {
            let _ = source_book_keys;
            Ok(SemanticRemoveResult {
                enabled: false,
                vectors_removed: 0,
            })
        }
    }

    /// Discard all sidecar vectors and manifest book entries. Lexical Tantivy
    /// documents are untouched, so a full semantic rebuild can follow safely.
    ///
    /// `&self` for the same reason as [`Self::semantic_index_books`].
    pub fn reset_semantic_index(&self) -> Result<SemanticResetResult> {
        #[cfg(feature = "semantic-integration")]
        {
            let Some(runtime) = self.semantic_engine() else {
                return Ok(SemanticResetResult {
                    enabled: false,
                    vectors_removed: 0,
                });
            };
            let removed = runtime
                .reset_semantic_index()
                .map_err(|err| anyhow::anyhow!("semantic reset failed: {err}"))?
                .unwrap_or(0);
            Ok(SemanticResetResult {
                enabled: true,
                vectors_removed: removed,
            })
        }

        #[cfg(not(feature = "semantic-integration"))]
        {
            Ok(SemanticResetResult {
                enabled: false,
                vectors_removed: 0,
            })
        }
    }

    /// Search through the sidecar exactly once. Tantivy supplies scored lexical
    /// candidates; `OtzariaHybridEngine` alone performs hybrid fusion/grouping.
    /// Semantic-only items are hydrated from Tantivy before crossing FFI.
    #[allow(clippy::too_many_arguments)]
    pub fn search_semantic(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        lexical_mode: SemanticLexicalMode,
        fuzzy_max_distance: u8,
        retrieval_mode: SemanticRetrievalMode,
        grouping: Option<SemanticGroupingMode>,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<SemanticSearchResponse> {
        let started = Instant::now();
        #[cfg(feature = "semantic-integration")]
        {
            let Some(runtime) = self.semantic_engine() else {
                return self.semantic_lexical_fallback_response(
                    &query,
                    &facets,
                    limit,
                    offset,
                    lexical_mode,
                    fuzzy_max_distance,
                    retrieval_mode,
                    grouping,
                    match_nikud,
                    match_taamim,
                    Some("semantic sidecar has not been configured".to_string()),
                    started.elapsed().as_millis() as u64,
                );
            };
            // Ask the coordinator for a prefix wider than the requested page,
            // then hydrate/filter before applying the caller's pagination.
            // This lets a few stale sidecar records be skipped without leaving
            // avoidable holes or shifting offsets between adjacent pages.
            let requested_window = offset.saturating_add(limit.saturating_mul(2)).max(1);
            let candidate_window_capped = requested_window > MAX_SEMANTIC_CANDIDATE_WINDOW;
            let candidate_window = requested_window.min(MAX_SEMANTIC_CANDIDATE_WINDOW);
            let SemanticLexicalPhase {
                candidates: lexical_candidates,
                total_count: lexical_total_count,
                truncated,
                highlight,
            } = if matches!(retrieval_mode, SemanticRetrievalMode::SemanticOnly) {
                // Semantic-only discards BM25 candidates in the coordinator.
                // Count lexically for the response envelope, but do not pay
                // to materialize and hydrate a TopDocs window that is unused.
                let count = match lexical_mode {
                    SemanticLexicalMode::Exact => self.count_exact_with_status(
                        query.clone(),
                        facets.clone(),
                        match_nikud,
                        match_taamim,
                    )?,
                    SemanticLexicalMode::Fuzzy => self.count_fuzzy_with_status(
                        query.clone(),
                        facets.clone(),
                        fuzzy_max_distance,
                        match_nikud,
                        match_taamim,
                    )?,
                };
                SemanticLexicalPhase {
                    candidates: Vec::new(),
                    total_count: count.count,
                    truncated: count.truncated,
                    highlight: None,
                }
            } else {
                match lexical_mode {
                    SemanticLexicalMode::Exact => self.semantic_exact_lexical_candidates(
                        &query,
                        &facets,
                        candidate_window,
                        match_nikud,
                        match_taamim,
                    )?,
                    SemanticLexicalMode::Fuzzy => self.semantic_fuzzy_lexical_candidates(
                        &query,
                        &facets,
                        candidate_window,
                        fuzzy_max_distance,
                        match_nikud,
                        match_taamim,
                    )?,
                }
            };
            let result = runtime
                .search(SidecarSearchRequest {
                    query,
                    lexical_candidates,
                    limit: Some(candidate_window),
                    offset: Some(0),
                    grouping: grouping.map(|value| match value {
                        SemanticGroupingMode::SameSection => SidecarGroupingMode::SameSection,
                        SemanticGroupingMode::IdenticalText => SidecarGroupingMode::IdenticalText,
                    }),
                    filters: Some(SidecarSearchFilters {
                        book_paths: None,
                        facets: (!facets.is_empty()).then_some(facets),
                        include_pdf: None,
                    }),
                    force_mode: Some(match retrieval_mode {
                        SemanticRetrievalMode::Hybrid => SidecarSearchMode::Hybrid,
                        SemanticRetrievalMode::SemanticOnly => SidecarSearchMode::SemanticOnly,
                        SemanticRetrievalMode::LexicalOnly => SidecarSearchMode::LexicalOnly,
                    }),
                    // Both are per-request overrides the sidecar added; `None` keeps its
                    // configured profile and flags, which is what this call has always
                    // used. Choosing either from here is S5's decision, not a repin's.
                    profile: None,
                    feature_flags: None,
                })
                .map_err(|err| anyhow::anyhow!("semantic search failed: {err}"))?;

            // Phase 1 — drop stale primaries across the whole window, keeping
            // the hydrated document so the surviving page needs no second
            // lookup. Only a `needs_hydration` item can be stale: a lexical
            // candidate came from this same searcher in this same request, so it
            // is live by construction and needs no existence check.
            //
            // This has to precede pagination, or a dropped record would leave a
            // hole on one page and shift the next.
            let mut surviving = Vec::with_capacity(result.results.len());
            let mut stale_primaries_dropped = 0u32;
            for item in result.results {
                if !item.needs_hydration {
                    surviving.push((item, None));
                    continue;
                }
                match self.get_document_by_id(item.id)? {
                    Some(document) => surviving.push((item, Some(document))),
                    // A semantic record whose Tantivy document disappeared is
                    // stale. Never send its old metadata to Dart: a failed or
                    // delayed sidecar cleanup must not resurrect deleted
                    // content.
                    None => stale_primaries_dropped = stale_primaries_dropped.saturating_add(1),
                }
            }

            // Phase 2 — paginate *before* the per-result work whose cost is
            // proportional to the page: sibling hydration is one Tantivy lookup
            // each and snippet painting tokenizes the line. Doing either for the
            // whole candidate window would waste work that grows linearly with
            // `offset`.
            let page: Vec<_> = surviving
                .into_iter()
                .skip(offset as usize)
                .take(limit as usize)
                .collect();

            let painter = match highlight {
                Some(highlight) if !page.is_empty() => {
                    Some(self.semantic_snippet_painter(highlight)?)
                }
                _ => None,
            };
            // Same budget the lexical API's default highlight uses, so both
            // paths bound the display string identically.
            let snippet_budget = HighlightConfig::default().max_chars;

            let mut results = Vec::with_capacity(page.len());
            let mut stale_siblings_dropped = 0u32;
            for (item, hydrated) in page {
                let original_merged_count = item.merged_count;
                let mut page_stale_siblings = 0u32;
                let mut merged = Vec::with_capacity(item.merged.len());
                for sibling in item.merged {
                    match self.get_document_by_id(sibling.id)? {
                        Some(document) => merged.push(MergedSibling {
                            title: document.title,
                            reference: document.reference,
                            id: document.id,
                            segment: document.segment,
                            is_pdf: document.is_pdf,
                            file_path: document.file_path,
                        }),
                        None => page_stale_siblings = page_stale_siblings.saturating_add(1),
                    }
                }
                stale_siblings_dropped = stale_siblings_dropped.saturating_add(page_stale_siblings);

                // Prefer Tantivy's copy for a hydrated item and move the fields
                // out of whichever record wins, rather than cloning them.
                let (title, reference, text, segment, is_pdf, file_path) = match hydrated {
                    Some(document) => (
                        document.title,
                        document.reference,
                        document.text,
                        document.segment,
                        document.is_pdf,
                        document.file_path,
                    ),
                    None => (
                        item.title,
                        item.reference,
                        item.text,
                        item.segment,
                        item.is_pdf,
                        item.file_path,
                    ),
                };
                let (snippet_html, is_highlighted) = match painter.as_ref() {
                    // A BM25 score is present exactly when Tantivy returned this
                    // line for the lexical query, which is what licenses the
                    // phrase-fallback term painting inside `paint`.
                    Some(painter) => painter.paint(&text, item.lexical_score.is_some()),
                    // `SemanticOnly` runs no lexical query, so there is nothing
                    // to paint with — the line still crosses FFI bounded.
                    None => (bounded_plain_snippet(&text, snippet_budget), false),
                };

                results.push(SemanticSearchResult {
                    title,
                    reference,
                    snippet_html,
                    is_highlighted,
                    id: item.id,
                    segment,
                    is_pdf,
                    file_path,
                    // The sidecar intentionally caps the materialized sibling
                    // list. Preserve its full group count and subtract only
                    // stale siblings that were actually observed in that list.
                    merged_count: original_merged_count
                        .saturating_sub(page_stale_siblings)
                        .max(1),
                    merged,
                    lexical_score: item.lexical_score,
                    semantic_score: item.semantic_score,
                    fused_score: item.fused_score,
                    source: match item.source {
                        SidecarResultSource::Lexical => SemanticResultSource::Lexical,
                        SidecarResultSource::Semantic => SemanticResultSource::Semantic,
                        SidecarResultSource::Both => SemanticResultSource::Both,
                    },
                    needs_hydration: false,
                });
            }
            let mut fallback_reason = result.fallback_reason;
            // Primaries and siblings are counted and reported separately: the
            // first are whole result cards removed from the candidate window,
            // the second are group members missing from the cards on this page
            // only — siblings are hydrated after pagination, so their count is
            // page-scoped while the primary count covers the window.
            if stale_primaries_dropped > 0 {
                let stale_reason = format!(
                    "dropped {stale_primaries_dropped} stale semantic result(s) missing from \
                     Tantivy; candidate counts may still include stale records; rebuild or \
                     reconcile the semantic index"
                );
                fallback_reason = Some(match fallback_reason {
                    Some(reason) => format!("{reason}; {stale_reason}"),
                    None => stale_reason,
                });
            }
            if stale_siblings_dropped > 0 {
                let stale_reason = format!(
                    "dropped {stale_siblings_dropped} stale grouped sibling(s) missing from \
                     Tantivy on this page; rebuild or reconcile the semantic index"
                );
                fallback_reason = Some(match fallback_reason {
                    Some(reason) => format!("{reason}; {stale_reason}"),
                    None => stale_reason,
                });
            }
            if candidate_window_capped {
                let cap_reason = format!(
                    "semantic candidate window capped at {MAX_SEMANTIC_CANDIDATE_WINDOW} \
                     (requested {requested_window})"
                );
                fallback_reason = Some(match fallback_reason {
                    Some(reason) => format!("{reason}; {cap_reason}"),
                    None => cap_reason,
                });
            }
            Ok(SemanticSearchResponse {
                results,
                // These remain stable across pages. They deliberately retain
                // the sidecar's candidate-set semantics instead of subtracting
                // only the stale records that happened to occur on this page.
                total_count: result.total_count,
                lexical_total_count,
                group_count: result.group_count,
                counts_are_exact: false,
                requested_mode: retrieval_mode,
                executed_mode: match result.search_mode {
                    SidecarSearchMode::Hybrid => SemanticExecutedMode::Hybrid,
                    SidecarSearchMode::SemanticOnly => SemanticExecutedMode::SemanticOnly,
                    SidecarSearchMode::LexicalOnly => SemanticExecutedMode::LexicalOnly,
                },
                semantic_available: result.semantic_available,
                fallback_reason,
                latency_ms: started.elapsed().as_millis() as u64,
                candidate_window_truncated: candidate_window_capped,
                truncated,
            })
        }

        #[cfg(not(feature = "semantic-integration"))]
        {
            self.semantic_lexical_fallback_response(
                &query,
                &facets,
                limit,
                offset,
                lexical_mode,
                fuzzy_max_distance,
                retrieval_mode,
                grouping,
                match_nikud,
                match_taamim,
                Some("semantic support is not compiled into this build".to_string()),
                started.elapsed().as_millis() as u64,
            )
        }
    }

    #[cfg(not(feature = "semantic-integration"))]
    fn semantic_disabled_status() -> SemanticStatus {
        SemanticStatus {
            enabled: false,
            available: false,
            model_loaded: false,
            indexed_book_count: 0,
            vector_count: 0,
            model_id: String::new(),
            embedding_dim: 0,
            embedding_backend: None,
            vector_backend: String::new(),
            vectors_persisted: false,
            needs_full_reindex: None,
            last_error: Some("semantic support is not compiled into this build".to_string()),
        }
    }

    #[cfg(feature = "semantic-integration")]
    fn semantic_not_configured_status() -> SemanticStatus {
        SemanticStatus {
            enabled: false,
            available: false,
            model_loaded: false,
            indexed_book_count: 0,
            vector_count: 0,
            model_id: String::new(),
            embedding_dim: 0,
            embedding_backend: None,
            vector_backend: String::new(),
            vectors_persisted: false,
            needs_full_reindex: None,
            last_error: Some("semantic sidecar has not been configured".to_string()),
        }
    }

    fn semantic_disabled_diff() -> SemanticIndexDiff {
        SemanticIndexDiff {
            enabled: false,
            new_books: Vec::new(),
            changed_books: Vec::new(),
            unverifiable_books: Vec::new(),
            removed_books: Vec::new(),
            model_mismatch: false,
            chunking_mismatch: false,
            normalization_mismatch: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn semantic_lexical_fallback_response(
        &self,
        query: &str,
        facets: &[String],
        limit: u32,
        offset: u32,
        lexical_mode: SemanticLexicalMode,
        fuzzy_max_distance: u8,
        requested_mode: SemanticRetrievalMode,
        grouping: Option<SemanticGroupingMode>,
        match_nikud: bool,
        match_taamim: bool,
        fallback_reason: Option<String>,
        latency_ms: u64,
    ) -> Result<SemanticSearchResponse> {
        if matches!(requested_mode, SemanticRetrievalMode::SemanticOnly) {
            let count = match lexical_mode {
                SemanticLexicalMode::Exact => self.count_exact_with_status(
                    query.to_string(),
                    facets.to_vec(),
                    match_nikud,
                    match_taamim,
                )?,
                SemanticLexicalMode::Fuzzy => self.count_fuzzy_with_status(
                    query.to_string(),
                    facets.to_vec(),
                    fuzzy_max_distance,
                    match_nikud,
                    match_taamim,
                )?,
            };
            return Ok(SemanticSearchResponse {
                results: Vec::new(),
                total_count: 0,
                lexical_total_count: count.count,
                group_count: None,
                counts_are_exact: true,
                requested_mode,
                executed_mode: SemanticExecutedMode::SemanticOnly,
                semantic_available: false,
                fallback_reason,
                latency_ms,
                candidate_window_truncated: false,
                truncated: count.truncated,
            });
        }

        let grouping = grouping.map(|value| match value {
            SemanticGroupingMode::SameSection => ResultGrouping::SameSection,
            SemanticGroupingMode::IdenticalText => ResultGrouping::IdenticalText,
        });
        let page = match lexical_mode {
            SemanticLexicalMode::Exact => self.search_and_count_exact(
                query.to_string(),
                facets.to_vec(),
                limit,
                offset,
                ResultsOrder::Relevance,
                match_nikud,
                match_taamim,
                grouping,
            )?,
            SemanticLexicalMode::Fuzzy => self.search_and_count_fuzzy(
                query.to_string(),
                facets.to_vec(),
                limit,
                offset,
                fuzzy_max_distance,
                ResultsOrder::Relevance,
                match_nikud,
                match_taamim,
                grouping,
            )?,
        };
        let hl = HighlightConfig::default();
        let results = page
            .results
            .into_iter()
            .enumerate()
            .map(|(rank, item)| {
                // `SearchResult::text` is snippet HTML when the highlighter
                // painted the line, and the raw stored line when it painted
                // nothing (a bare fuzzy automaton exposes no static terms to the
                // generator). Only the latter needs escaping and bounding, so
                // the semantic envelope carries the same kind of value on both
                // paths. Detection is by the prefix the highlighter inserts:
                // painted output has every literal `<` escaped, so the marker
                // can only be real markup — and a corpus line that happens to
                // contain it verbatim is passed through exactly as the existing
                // lexical API already passes it through.
                let is_highlighted = item.text.contains(&hl.highlight_prefix);
                let snippet_html = if is_highlighted {
                    item.text
                } else {
                    bounded_plain_snippet(&item.text, hl.max_chars)
                };
                SemanticSearchResult {
                    title: item.title,
                    reference: item.reference,
                    snippet_html,
                    is_highlighted,
                    id: item.id,
                    segment: item.segment,
                    is_pdf: item.is_pdf,
                    file_path: item.file_path,
                    merged_count: item.merged_count,
                    merged: item.merged,
                    lexical_score: None,
                    semantic_score: None,
                    // Tantivy's legacy display API does not expose its score.
                    // Keep the existing relevance order observable without
                    // pretending a synthetic value is a BM25 score.
                    fused_score: 1.0 / (rank.saturating_add(1) as f32),
                    source: SemanticResultSource::Lexical,
                    needs_hydration: false,
                }
            })
            .collect();
        Ok(SemanticSearchResponse {
            total_count: page.total_count,
            lexical_total_count: page.total_count,
            group_count: page.group_count,
            counts_are_exact: !page.truncated,
            results,
            requested_mode,
            executed_mode: SemanticExecutedMode::LexicalOnly,
            semantic_available: false,
            fallback_reason,
            latency_ms,
            candidate_window_truncated: false,
            truncated: page.truncated,
        })
    }

    // ── Write API ──────────────────────────────────────────────────────────────

    /// Add a single document. Does not commit.
    /// Writes no content fingerprint (`contentHash` = 0) — batch ingestion via
    /// [`Self::add_documents_batch`] is the fingerprint-aware path.
    ///
    /// `_text` is normalized here ([`normalize_text_for_indexing`]) — the API
    /// is exposed over FFI, so the "input is already normalized" assumption
    /// is enforced rather than documented. Already-normalized text passes
    /// through the fast paths at negligible cost.
    pub fn add_document(
        &mut self,
        _id: u64,
        _title: &str,
        _reference: &str,
        _topics: &str,
        _text: &str,
        _segment: u64,
        _is_pdf: bool,
        _file_path: &str,
        _section_id: Option<u64>,
        _generation_order: Option<u32>,
        _extra_facets: Option<Vec<String>>,
    ) -> Result<()> {
        let (
            title_f,
            reference_f,
            text_f,
            id_f,
            segment_f,
            is_pdf_f,
            file_path_f,
            topics_f,
            content_hash_f,
            text_hash_f,
            text_vocalized_f,
            section_id_f,
            generation_sort_f,
            line_hash_f,
        ) = self.all_fields()?;
        let topics_facet = Facet::from_text(_topics)?;
        let normalized_text = hebrew_query::normalize_text_for_indexing(_text);
        let mut document = doc!(
            title_f        => _title,
            reference_f    => _reference,
            text_f         => normalized_text.as_str(),
            id_f           => _id,
            segment_f      => _segment,
            is_pdf_f       => _is_pdf,
            file_path_f    => _file_path,
            topics_f       => topics_facet,
            content_hash_f => 0u64,
            text_hash_f    => 0u64,
            section_id_f   => _section_id.unwrap_or(_id),
            generation_sort_f => generation_sort_key(
                _generation_order.unwrap_or(DEFAULT_GENERATION_ORDER),
                _id
            ),
            line_hash_f    => line_dedup_hash(&normalized_text)
        );
        for facet in _extra_facets.iter().flatten() {
            document.add_facet(topics_f, Facet::from_text(facet)?);
        }
        // שורה שנושאת סימנים משתתפת גם בחיפוש המנוקד; הבדיקה על הקלט הגולמי
        // (הנרמול הרגיל מסיר את הסימנים) והעותק המנוקד מנורמל בנפרד.
        if hebrew_query::contains_attached_marks(_text) {
            document.add_text(
                text_vocalized_f,
                hebrew_query::normalize_vocalized_text_for_indexing(_text),
            );
        }
        self.writer_mut()?.add_document(document)?;
        Ok(())
    }

    /// Add many documents in a single FFI call. Does not commit.
    /// For initial bulk loads – no duplicate checking.
    ///
    /// `text` is normalized here ([`normalize_text_for_indexing`]) and a
    /// supplied `text_vocalized` through its vocalized counterpart — the API
    /// is exposed over FFI, so the "input is already normalized" assumption
    /// is enforced rather than documented. Already-normalized input passes
    /// through the fast paths at negligible cost.
    pub fn add_documents_batch(&mut self, docs: Vec<DocumentInput>) -> Result<()> {
        let (
            title_f,
            reference_f,
            text_f,
            id_f,
            segment_f,
            is_pdf_f,
            file_path_f,
            topics_f,
            content_hash_f,
            text_hash_f,
            text_vocalized_f,
            section_id_f,
            generation_sort_f,
            line_hash_f,
        ) = self.all_fields()?;
        let writer = self.writer_mut()?;
        for doc in docs {
            let topics_facet = Facet::from_text(&doc.topics)?;
            let normalized_text = hebrew_query::normalize_text_for_indexing(&doc.text);
            let line_hash = line_dedup_hash(&normalized_text);
            let mut document = doc!(
                title_f        => doc.title,
                reference_f    => doc.reference,
                text_f         => normalized_text,
                id_f           => doc.id,
                segment_f      => doc.segment,
                is_pdf_f       => doc.is_pdf,
                file_path_f    => doc.file_path,
                topics_f       => topics_facet,
                content_hash_f => doc.content_hash.unwrap_or(0),
                text_hash_f    => doc.text_hash.unwrap_or(0),
                section_id_f   => doc.section_id.unwrap_or(doc.id),
                generation_sort_f => generation_sort_key(
                    doc.generation_order.unwrap_or(DEFAULT_GENERATION_ORDER),
                    doc.id
                ),
                line_hash_f    => line_hash
            );
            for facet in doc.extra_facets.iter().flatten() {
                document.add_facet(topics_f, Facet::from_text(facet)?);
            }
            if let Some(vocalized) = doc.text_vocalized {
                if !vocalized.is_empty() {
                    document.add_text(
                        text_vocalized_f,
                        hebrew_query::normalize_vocalized_text_for_indexing(&vocalized),
                    );
                }
            }
            writer.add_document(document)?;
        }
        Ok(())
    }

    /// Indexes a whole text book in ONE FFI call. Does not commit.
    ///
    /// Splits `text` into lines, tracks the `<h…>` heading reference trail,
    /// normalizes each line ([`normalize_text_for_indexing`]), stamps the
    /// book's canonical fingerprint ([`compute_book_fingerprint`] — text +
    /// metadata) on every document, and adds one document per line. Returns the number of documents added (0 for empty
    /// text — the caller writes its empty-book marker in that case).
    ///
    /// This is the whole-book replacement for the app's per-line pipeline
    /// (Dart isolate → per-batch FFI normalize → SendPort copy → batch add):
    /// the raw text crosses the bridge exactly once and only a count comes
    /// back. Document ids encode catalogue order exactly like the Dart
    /// `buildCatalogueDocumentId`: `((catalogue_order+1) << 32) + ordinal+1`.
    pub fn add_text_book(
        &mut self,
        title: String,
        topics: String,
        file_path: String,
        catalogue_order: u32,
        generation_order: u32,
        text: String,
        extra_facets: Option<Vec<String>>,
    ) -> Result<u32> {
        self.add_text_book_impl(
            title,
            topics,
            file_path,
            catalogue_order,
            generation_order,
            &text,
            extra_facets.unwrap_or_default(),
        )
    }

    /// [`Self::add_text_book`] over raw UTF-8 bytes. The app reads book
    /// content from SQLite, which stores UTF-8 — passing the bytes through
    /// (SQLite BLOB → `Uint8List` → here) skips the UTF-8→UTF-16→UTF-8
    /// round-trip a Dart `String` costs on the bridge (~180ms/MB measured).
    /// Invalid UTF-8 is replaced (lossy), never an error — matching what the
    /// Dart decode would have produced.
    pub fn add_text_book_bytes(
        &mut self,
        title: String,
        topics: String,
        file_path: String,
        catalogue_order: u32,
        generation_order: u32,
        text: Vec<u8>,
        extra_facets: Option<Vec<String>>,
    ) -> Result<u32> {
        let text = match String::from_utf8_lossy(&text) {
            std::borrow::Cow::Borrowed(_) => {
                // Valid UTF-8 — take ownership without re-copying.
                unsafe { String::from_utf8_unchecked(text) }
            }
            std::borrow::Cow::Owned(fixed) => fixed,
        };
        self.add_text_book_impl(
            title,
            topics,
            file_path,
            catalogue_order,
            generation_order,
            &text,
            extra_facets.unwrap_or_default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_text_book_impl(
        &mut self,
        title: String,
        topics: String,
        file_path: String,
        catalogue_order: u32,
        generation_order: u32,
        text: &str,
        extra_facets: Vec<String>,
    ) -> Result<u32> {
        if text.is_empty() {
            return Ok(0);
        }
        let started = Instant::now();
        let text_bytes = text.len();
        let (
            title_f,
            reference_f,
            text_f,
            id_f,
            segment_f,
            is_pdf_f,
            file_path_f,
            topics_f,
            content_hash_f,
            text_hash_f,
            text_vocalized_f,
            section_id_f,
            generation_sort_f,
            line_hash_f,
        ) = self.all_fields()?;
        let topics_facet = Facet::from_text(&topics)?;
        // נתיבי הממדים (מחבר/תקופה/יסוד) נפרסים פעם אחת לספר ומוטבעים
        // כערכי facet נוספים על כל שורה.
        let extra_facet_values: Vec<Facet> = extra_facets
            .iter()
            .map(|f| Facet::from_text(f))
            .collect::<std::result::Result<_, _>>()?;
        let content_hash = book_fingerprint(
            text,
            &title,
            &topics,
            catalogue_order,
            generation_order,
            extra_facets.clone(),
        );
        let text_hash = content_fingerprint(text);
        let id_base = catalogue_id_base(catalogue_order)?;
        let writer = self.writer_mut()?;

        // "prepare" — the pure-CPU phase (trail + normalization); "enqueue" —
        // writer.add_document (queue push; grows only when tantivy's indexing
        // threads apply backpressure).
        let prepare_started = Instant::now();
        let lines: Vec<&str> = text.split('\n').collect();

        // Sequential cheap pass: the reference trail is stateful across
        // lines, so resolve each line to its reference index first. The
        // stripped trail is recomputed only when a heading changes it.
        let mut trail: Vec<&str> = Vec::new();
        let mut references: Vec<String> = vec![String::new()];
        let mut reference_of_line: Vec<u32> = Vec::with_capacity(lines.len());
        for raw_line in &lines {
            if raw_line.starts_with("<h") {
                update_reference_trail(&mut trail, raw_line);
                // כיווץ אחרי ההסרה: תג הכותרת עצמו (תג שבירה) הופך לרווח.
                references.push(
                    trail
                        .iter()
                        .map(|part| {
                            hebrew_query::collapse_whitespace(
                                &hebrew_query::strip_html_for_indexing(part),
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", "),
                );
            }
            reference_of_line.push((references.len() - 1) as u32);
        }

        // The expensive pass — normalization is order-independent across
        // lines, so it fans out over all cores. A line carrying nikud/taamim
        // also gets its vocalized rendering, for the `textVocalized` field
        // (the mark check is cheap and almost always short-circuits false).
        use rayon::prelude::*;
        let normalized: Vec<(String, Option<String>, u64)> = lines
            .par_iter()
            .map(|raw_line| {
                let plain = hebrew_query::normalize_text_for_indexing(raw_line);
                let vocalized = hebrew_query::contains_attached_marks(raw_line)
                    .then(|| hebrew_query::normalize_vocalized_text_for_indexing(raw_line))
                    .filter(|v| !v.is_empty());
                let line_hash = line_dedup_hash(&plain);
                (plain, vocalized, line_hash)
            })
            .collect();
        let prepare_time = prepare_started.elapsed();

        let enqueue_started = Instant::now();
        let mut ordinal: u64 = 0;
        for (segment, (normalized_line, vocalized_line, line_hash)) in
            normalized.into_iter().enumerate()
        {
            let reference = references[reference_of_line[segment] as usize].as_str();
            let id = id_base + ordinal + 1;
            let mut document = doc!(
                title_f        => title.as_str(),
                reference_f    => reference,
                text_f         => normalized_line,
                id_f           => id,
                segment_f      => segment as u64,
                is_pdf_f       => false,
                file_path_f    => file_path.as_str(),
                topics_f       => topics_facet.clone(),
                content_hash_f => content_hash,
                text_hash_f    => text_hash,
                // כל השורות של אותו בלוק כותרת חולקות ערך; id_base מבדל
                // בין ספרים, אז המזהה ייחודי גלובלית.
                section_id_f   => id_base + u64::from(reference_of_line[segment]),
                generation_sort_f => generation_sort_key(generation_order, id),
                line_hash_f    => line_hash
            );
            for facet in &extra_facet_values {
                document.add_facet(topics_f, facet.clone());
            }
            if let Some(vocalized) = vocalized_line {
                document.add_text(text_vocalized_f, vocalized);
            }
            writer.add_document(document)?;
            ordinal += 1;
        }
        let enqueue_time = enqueue_started.elapsed();
        info!(
            "add_text_book '{title}': {ordinal} docs, {text_bytes} bytes in {:?} \
             (prepare {prepare_time:?}, enqueue {enqueue_time:?})",
            started.elapsed()
        );
        Ok(ordinal as u32)
    }

    /// Indexes a whole PDF book in ONE FFI call. Does not commit.
    ///
    /// The whole-book replacement for the app's per-page PDF pipeline (Dart
    /// isolate → per-window FFI normalize → SendPort copy → batch add), which
    /// copied the extracted text four-five times per book. Each page's text is
    /// split into lines; every line is normalised
    /// ([`normalize_pdf_text_for_indexing`]) and dropped when the garbage
    /// heuristic ([`is_probably_garbage_pdf_text`]) flags it — exactly the
    /// per-line logic of [`normalize_pdf_texts_for_indexing`]. One document is
    /// added per surviving line, with `segment` = the page's `page_index` and
    /// ids encoding catalogue order like [`Self::add_text_book`]. PDFs record
    /// no content fingerprint (`contentHash` = 0, their text is not in the
    /// library DB).
    ///
    /// Returns the number of documents added; 0 means the PDF yielded no
    /// usable text (scanned/garbage) — the caller falls back to a sidecar or
    /// writes its empty-book marker.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pdf_book(
        &mut self,
        title: String,
        topics: String,
        file_path: String,
        catalogue_order: u32,
        generation_order: u32,
        pages: Vec<PdfPageInput>,
        extra_facets: Option<Vec<String>>,
    ) -> Result<u32> {
        if pages.is_empty() {
            return Ok(0);
        }
        let started = Instant::now();
        let text_bytes: usize = pages.iter().map(|p| p.text.len()).sum();
        let page_count = pages.len();
        let (
            title_f,
            reference_f,
            text_f,
            id_f,
            segment_f,
            is_pdf_f,
            file_path_f,
            topics_f,
            content_hash_f,
            text_hash_f,
            // PDF אינו משתתף בחיפוש מנוקד: ניקוד שמגיע מ-OCR אינו אמין,
            // והנרמול של PDF ממילא מוחק אותו.
            _text_vocalized_f,
            section_id_f,
            generation_sort_f,
            line_hash_f,
        ) = self.all_fields()?;
        let topics_facet = Facet::from_text(&topics)?;
        let extra_facet_values: Vec<Facet> = extra_facets
            .unwrap_or_default()
            .iter()
            .map(|f| Facet::from_text(f))
            .collect::<std::result::Result<_, _>>()?;
        let id_base = catalogue_id_base(catalogue_order)?;
        let writer = self.writer_mut()?;

        // Normalization + garbage heuristic are per-line pure functions —
        // fan the whole book out over all cores, then enqueue sequentially.
        use rayon::prelude::*;
        let prepare_started = Instant::now();
        let lines: Vec<(usize, &str)> = pages
            .iter()
            .enumerate()
            .flat_map(|(page_idx, page)| page.text.split('\n').map(move |line| (page_idx, line)))
            .collect();
        let prepared: Vec<(usize, String, bool)> = lines
            .par_iter()
            .map(|(page_idx, raw_line)| {
                let normalized = hebrew_query::normalize_pdf_text_for_indexing(raw_line);
                let is_garbage = hebrew_query::is_probably_garbage_pdf_text(&normalized);
                (*page_idx, normalized, is_garbage)
            })
            .collect();
        let prepare_time = prepare_started.elapsed();

        let enqueue_started = Instant::now();
        let mut ordinal: u64 = 0;
        let mut garbage_lines: u64 = 0;
        for (page_idx, normalized, is_garbage) in prepared {
            if is_garbage {
                garbage_lines += 1;
                continue;
            }
            let page = &pages[page_idx];
            let id = id_base + ordinal + 1;
            let line_hash = line_dedup_hash(&normalized);
            let mut document = doc!(
                title_f        => title.as_str(),
                reference_f    => page.reference.as_str(),
                text_f         => normalized,
                id_f           => id,
                segment_f      => u64::from(page.page_index),
                is_pdf_f       => true,
                file_path_f    => file_path.as_str(),
                topics_f       => topics_facet.clone(),
                content_hash_f => 0u64,
                text_hash_f    => 0u64,
                // ב-PDF אין שרשרת כותרות — עמוד = סעיף.
                section_id_f   => id_base + u64::from(page.page_index),
                generation_sort_f => generation_sort_key(generation_order, id),
                line_hash_f    => line_hash
            );
            for facet in &extra_facet_values {
                document.add_facet(topics_f, facet.clone());
            }
            writer.add_document(document)?;
            ordinal += 1;
        }
        let enqueue_time = enqueue_started.elapsed();
        info!(
            "add_pdf_book '{title}': {ordinal} docs from {page_count} pages \
             ({garbage_lines} garbage lines), {text_bytes} bytes in {:?} \
             (prepare {prepare_time:?}, enqueue {enqueue_time:?})",
            started.elapsed()
        );
        Ok(ordinal as u32)
    }

    /// Delete then re-insert a single document by id. Does not commit.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_document(
        &mut self,
        _id: u64,
        _title: &str,
        _reference: &str,
        _topics: &str,
        _text: &str,
        _segment: u64,
        _is_pdf: bool,
        _file_path: &str,
        _section_id: Option<u64>,
        _generation_order: Option<u32>,
        _extra_facets: Option<Vec<String>>,
    ) -> Result<()> {
        self.delete_document_by_id(_id)?;
        self.add_document(
            _id,
            _title,
            _reference,
            _topics,
            _text,
            _segment,
            _is_pdf,
            _file_path,
            _section_id,
            _generation_order,
            _extra_facets,
        )
    }

    /// Upsert many documents in a single FFI call. Does not commit.
    /// Normalizes `text`/`text_vocalized` exactly like
    /// [`Self::add_documents_batch`].
    pub fn upsert_documents_batch(&mut self, docs: Vec<DocumentInput>) -> Result<()> {
        let (
            title_f,
            reference_f,
            text_f,
            id_f,
            segment_f,
            is_pdf_f,
            file_path_f,
            topics_f,
            content_hash_f,
            text_hash_f,
            text_vocalized_f,
            section_id_f,
            generation_sort_f,
            line_hash_f,
        ) = self.all_fields()?;
        let writer = self.writer_mut()?;
        for doc in docs {
            writer.delete_term(Term::from_field_u64(id_f, doc.id));
            let topics_facet = Facet::from_text(&doc.topics)?;
            let normalized_text = hebrew_query::normalize_text_for_indexing(&doc.text);
            let line_hash = line_dedup_hash(&normalized_text);
            let mut document = doc!(
                title_f        => doc.title,
                reference_f    => doc.reference,
                text_f         => normalized_text,
                id_f           => doc.id,
                segment_f      => doc.segment,
                is_pdf_f       => doc.is_pdf,
                file_path_f    => doc.file_path,
                topics_f       => topics_facet,
                content_hash_f => doc.content_hash.unwrap_or(0),
                text_hash_f    => doc.text_hash.unwrap_or(0),
                section_id_f   => doc.section_id.unwrap_or(doc.id),
                generation_sort_f => generation_sort_key(
                    doc.generation_order.unwrap_or(DEFAULT_GENERATION_ORDER),
                    doc.id
                ),
                line_hash_f    => line_hash
            );
            for facet in doc.extra_facets.iter().flatten() {
                document.add_facet(topics_f, Facet::from_text(facet)?);
            }
            if let Some(vocalized) = doc.text_vocalized {
                if !vocalized.is_empty() {
                    document.add_text(
                        text_vocalized_f,
                        hebrew_query::normalize_vocalized_text_for_indexing(&vocalized),
                    );
                }
            }
            writer.add_document(document)?;
        }
        Ok(())
    }

    /// Delete a document by its numeric id. Does not commit.
    /// Whether the index this engine opened is one this build reads.
    ///
    /// `pub(crate)`: the FFI already exposes [`check_index_compatibility`] by path. This is
    /// the same answer for the directory already open, so a build does not have to be told
    /// again where it is.
    #[cfg(feature = "semantic-integration")]
    pub(crate) fn index_compatibility(&self) -> IndexCompatibility {
        check_index_compatibility_path(&self.index_path)
    }

    /// A snapshot of the index for the semantic builder to read the corpus from.
    ///
    /// `pub(crate)`, so flutter_rust_bridge never sees it: Dart has no use for a
    /// `Searcher`, and the corpus port is a Rust-to-Rust contract with the sidecar. One
    /// call is one snapshot — see [`TantivyCorpus`](crate::semantic_corpus::TantivyCorpus),
    /// which holds it for a whole build precisely so a reload cannot move the corpus
    /// underneath one.
    #[cfg(feature = "semantic-integration")]
    pub(crate) fn corpus_searcher(&self) -> Searcher {
        self.index_reader.searcher()
    }

    /// Delete a document by its numeric id. Does not commit.
    pub fn delete_document_by_id(&mut self, id: u64) -> Result<()> {
        let id_f = self.schema.get_field("id").unwrap();
        self.writer_mut()?
            .delete_term(Term::from_field_u64(id_f, id));
        Ok(())
    }

    /// Delete every document of one book, addressed by its `filePath` value —
    /// the stable book key the app stamps on all of a book's documents at
    /// indexing time. `filePath` is a raw STRING field, so this is a single
    /// exact `delete_term` and cannot touch other books (unlike
    /// [`Self::remove_documents_by_title`], which matches any book sharing the
    /// title). Does not commit.
    pub fn delete_documents_by_file_path(&mut self, file_path: &str) -> Result<()> {
        let file_path_f = self.schema.get_field("filePath")?;
        self.writer_mut()?
            .delete_term(Term::from_field_text(file_path_f, file_path));
        Ok(())
    }

    /// Batch form of [`Self::delete_documents_by_file_path`] — one FFI call
    /// for e.g. removing a whole custom folder of personal books. Does not
    /// commit.
    pub fn delete_documents_by_file_paths(&mut self, file_paths: Vec<String>) -> Result<()> {
        let file_path_f = self.schema.get_field("filePath")?;
        let writer = self.writer_mut()?;
        for path in file_paths {
            writer.delete_term(Term::from_field_text(file_path_f, &path));
        }
        Ok(())
    }

    /// Delete all documents matching a title. Does not commit.
    /// Kept for backward compatibility – prefer delete_document_by_id.
    pub fn remove_documents_by_title(&mut self, title: &str) -> Result<()> {
        let title_field = self.schema.get_field("title")?;
        self.writer_mut()?
            .delete_term(Term::from_field_text(title_field, title));
        Ok(())
    }

    /// Delete all documents. Does not commit.
    pub fn clear(&mut self) -> Result<()> {
        self.writer_mut()?.delete_all_documents()?;
        Ok(())
    }

    /// Bulk-indexing mode: while enabled, the live writer skips background
    /// segment merges (`NoMergePolicy`). During a full-library build the
    /// default `LogMergePolicy` repeatedly merges intermediate segments —
    /// CPU and IO that are thrown away, because the caller runs `optimize`
    /// (merge-all) once at the end anyway. Call with `true` before a bulk
    /// build and `false` when done — `optimize` does NOT reset the flag, and
    /// while it is set every (re)opened writer keeps `NoMergePolicy`. Off by
    /// default; incremental indexing keeps normal merging.
    pub fn set_bulk_indexing(&mut self, enabled: bool) -> Result<()> {
        self.bulk_indexing = enabled;
        let writer = self.writer_mut()?;
        if enabled {
            writer.set_merge_policy(Box::new(NoMergePolicy));
        } else {
            writer.set_merge_policy(Box::<tantivy::indexer::LogMergePolicy>::default());
        }
        debug!("bulk_indexing={enabled}");
        Ok(())
    }

    /// Economy indexing mode: shrinks the writer's memory budget to the
    /// mobile footprint (50MB ⇒ tantivy caps itself at 3 indexing threads)
    /// so the machine stays responsive during a long build; `false` restores
    /// the platform default. The budget is fixed at writer creation, so a
    /// live writer is swapped — pending documents are committed first and
    /// nothing is lost. May be toggled mid-indexing; off by default.
    pub fn set_economy_indexing(&mut self, enabled: bool) -> Result<()> {
        let target = if enabled {
            ECONOMY_WRITER_HEAP_SIZE
        } else {
            DEFAULT_WRITER_HEAP_SIZE
        };
        if target == self.writer_heap_size {
            return Ok(());
        }
        self.writer_heap_size = target;
        if let Some(mut writer) = self.index_writer.take() {
            writer.commit()?;
            writer.wait_merging_threads()?;
            self.restore_writer()?;
            self.index_reader.reload()?;
        }
        debug!("economy_indexing={enabled} (writer budget {target} bytes)");
        Ok(())
    }

    /// Flush pending writes to disk and refresh the reader.
    pub fn commit(&mut self) -> Result<()> {
        let started = Instant::now();
        self.writer_mut()?.commit()?;
        let commit_elapsed = started.elapsed();
        let reload_started = Instant::now();
        self.index_reader.reload()?;
        info!(
            "commit: {commit_elapsed:?} (reader reload {:?})",
            reload_started.elapsed()
        );
        Ok(())
    }

    /// Discard all pending writes since the last commit.
    pub fn rollback(&mut self) -> Result<()> {
        self.writer_mut()?.rollback()?;
        Ok(())
    }

    // ── Search API ─────────────────────────────────────────────────────────────

    /// Paged regex search. Drops the single-word truncation flag: a broad
    /// query (e.g. `.*ספר`) that overflows its collection budget serves
    /// partial results with no signal. Not suitable for UI that must tell the
    /// user the result is partial — use [`Self::search_and_count`]
    /// ([`SearchPageResult::truncated`]) instead.
    pub fn search(
        &self,
        regex_terms: Vec<String>,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        slop: u32,
        max_expansions: u32,
        order: ResultsOrder,
        highlight: Option<HighlightConfig>,
    ) -> Result<Vec<SearchResult>> {
        let (query, _) = self.build_query(regex_terms, facets, slop, max_expansions)?;
        let hl = highlight.unwrap_or_else(HighlightConfig::default);
        self.run_search(
            query,
            |_| Ok(HighlightPlan::none()),
            self.schema.get_field("text")?,
            limit,
            offset,
            &order,
            &hl,
            None,
        )
    }

    /// Search and return total hit count alongside paged results in one call.
    /// Uses a tuple collector so Tantivy executes a single index pass.
    pub fn search_and_count(
        &self,
        regex_terms: Vec<String>,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        slop: u32,
        max_expansions: u32,
        order: ResultsOrder,
        highlight: Option<HighlightConfig>,
    ) -> Result<SearchPageResult> {
        let (query, truncated) = self.build_query(regex_terms, facets, slop, max_expansions)?;
        let hl = highlight.unwrap_or_else(HighlightConfig::default);
        self.run_search_and_count(
            query,
            |_| Ok(HighlightPlan::none()),
            self.schema.get_field("text")?,
            limit,
            offset,
            &order,
            &hl,
            truncated,
            None,
        )
    }

    /// Bare hit count. Drops the single-word truncation flag: a broad query
    /// (e.g. `.*ספר`) that overflows its collection budget returns a partial
    /// count with no signal. Not suitable for UI that must tell the user the
    /// result is partial — use [`Self::count_with_status`], the combined
    /// stream, or [`SearchPageResult::truncated`] there instead.
    pub fn count(
        &self,
        regex_terms: Vec<String>,
        facets: &[String],
        slop: u32,
        max_expansions: u32,
    ) -> Result<u32> {
        Ok(self
            .count_with_status(regex_terms, facets, slop, max_expansions)?
            .count)
    }

    /// Like [`Self::count`] but also reports whether single-word collection
    /// truncated, so a UI consumer can flag a partial count.
    pub fn count_with_status(
        &self,
        regex_terms: Vec<String>,
        facets: &[String],
        slop: u32,
        max_expansions: u32,
    ) -> Result<CountResult> {
        let (query, truncated) =
            self.build_query(regex_terms, facets.to_vec(), slop, max_expansions)?;
        Ok(CountResult {
            count: self.run_count(query)?,
            truncated,
        })
    }

    /// Per-book hit counts. Drops the truncation flag — see [`Self::count`];
    /// use [`Self::count_by_book_with_status`] when partiality must surface.
    pub fn count_by_book(
        &self,
        regex_terms: Vec<String>,
        facets: Vec<String>,
        slop: u32,
        max_expansions: u32,
    ) -> Result<HashMap<String, u32>> {
        Ok(self
            .count_by_book_with_status(regex_terms, facets, slop, max_expansions)?
            .counts)
    }

    /// Like [`Self::count_by_book`] but also reports single-word truncation.
    pub fn count_by_book_with_status(
        &self,
        regex_terms: Vec<String>,
        facets: Vec<String>,
        slop: u32,
        max_expansions: u32,
    ) -> Result<BookCountResult> {
        let (query, truncated) = self.build_query(regex_terms, facets, slop, max_expansions)?;
        Ok(BookCountResult {
            counts: self.run_count_by_book(query)?,
            truncated,
        })
    }

    /// Return per-child facet counts for a given prefix (e.g. "/"). Drops the
    /// truncation flag — see [`Self::count`]; use
    /// [`Self::get_facet_counts_with_status`] when partiality must surface.
    ///
    /// שימו לב: ממדי הסינון חיים באותו שדה facet כמו עץ הקטגוריות, ולכן
    /// תחת prefix `/` מופיעים גם השורשים השמורים [`FACET_DIMENSION_ROOTS`]
    /// (`/author`, `/era`, `/base`) לצד קטגוריות-העל. לקוח שמונה ילדים
    /// כדי לבנות עץ קטגוריות חייב לסנן אותם (באפליקציה:
    /// `FacetHelper.isDimensionFacet`); לקוח שקורא ספירות לפי נתיבים
    /// ידועים מראש אינו מושפע.
    pub fn get_facet_counts(
        &self,
        regex_terms: Vec<String>,
        facets: Vec<String>,
        facet_prefix: String,
        slop: u32,
        max_expansions: u32,
    ) -> Result<Vec<FacetCount>> {
        Ok(self
            .get_facet_counts_with_status(regex_terms, facets, facet_prefix, slop, max_expansions)?
            .counts)
    }

    /// Like [`Self::get_facet_counts`] but also reports single-word truncation.
    pub fn get_facet_counts_with_status(
        &self,
        regex_terms: Vec<String>,
        facets: Vec<String>,
        facet_prefix: String,
        slop: u32,
        max_expansions: u32,
    ) -> Result<FacetCountsResult> {
        let (query, truncated) = self.build_query(regex_terms, facets, slop, max_expansions)?;
        Ok(FacetCountsResult {
            counts: self.run_facet_counts(query, facet_prefix)?,
            truncated,
        })
    }

    // ── Operational API ────────────────────────────────────────────────────────

    /// Compact the index without collapsing it. Pending changes are committed
    /// first (only committed segments take part in manual merges); then the
    /// searchable segments are trimmed to at most `MAX_SEGMENTS_AFTER_OPTIMIZE`
    /// by merging the *smallest* ones together, and every segment whose
    /// deleted-doc share exceeds `OPTIMIZE_COMPACT_DELETE_RATIO` joins that
    /// merge so its space is reclaimed. Large healthy segments are never
    /// rewritten — tantivy searches segments in parallel, so a single segment
    /// buys nothing — which keeps the cost of an incremental run proportional
    /// to what it added rather than to the size of the whole index.
    pub fn optimize(&mut self) -> Result<()> {
        let started = Instant::now();
        let before_count = self.index.searchable_segment_ids()?.len();
        debug!("optimize: before={before_count}");
        if before_count == 0 {
            debug!("optimize: skipped");
            return Ok(());
        }

        let mut writer = self.take_writer()?;
        let maintenance_result = (|| -> Result<()> {
            // Dropping the writer discards its RAM buffer; flush pending
            // changes first so optimize never silently loses documents.
            writer.commit()?;
            writer.wait_merging_threads()?;
            self.optimize_committed_segments()
        })();
        let restore_result = self.restore_writer();

        if let Err(restore_err) = restore_result {
            return match maintenance_result {
                Ok(_) => Err(restore_err),
                Err(maintenance_err) => Err(restore_err.context(format!(
                    "optimize maintenance also failed: {maintenance_err:#}"
                ))),
            };
        }

        maintenance_result?;
        self.index_reader.reload()?;
        self.collect_garbage_after_optimize();
        let after_count = self.index.searchable_segment_ids()?.len();
        info!(
            "optimize: {before_count} → {after_count} segments in {:?}",
            started.elapsed()
        );
        Ok(())
    }

    /// Tantivy's post-merge GC keeps merged-away segments while the old searcher
    /// holds their metas/mmaps; run it again after `reload` released them.
    fn collect_garbage_after_optimize(&mut self) {
        let gc = match self.writer_mut() {
            Ok(writer) => writer.garbage_collect_files().wait(),
            Err(err) => {
                warn!("optimize: writer unavailable for garbage collection: {err:#}");
                return;
            }
        };
        match gc {
            Ok(result) => debug!(
                "optimize: garbage collection deleted {} files ({} could not be deleted)",
                result.deleted_files.len(),
                result.failed_to_delete_files.len()
            ),
            Err(err) => warn!("optimize: garbage collection failed: {err}"),
        }
    }

    pub fn get_document_count(&self) -> u64 {
        self.index_reader.searcher().num_docs()
    }

    pub fn get_segment_count(&self) -> Result<u32> {
        Ok(self.index.searchable_segment_ids()?.len() as u32)
    }

    /// Number of live (committed, non-deleted) documents per distinct
    /// `filePath` across the whole index.
    ///
    /// This is read from the index itself rather than from any external state,
    /// so callers can reconstruct indexing progress directly from an index —
    /// e.g. after pointing the engine at a directory that already contains an
    /// index built elsewhere — and compare it against the current library.
    pub fn count_documents_by_file_path(&self) -> Result<HashMap<String, u32>> {
        let searcher = self.index_reader.searcher();
        Ok(searcher.search(&AllQuery, &BookCountCollector)?)
    }

    /// Distinct `filePath` values present in the index — i.e. which books have
    /// at least one live document. Convenience wrapper over
    /// [`Self::count_documents_by_file_path`].
    pub fn get_indexed_file_paths(&self) -> Result<Vec<String>> {
        Ok(self.count_documents_by_file_path()?.into_keys().collect())
    }

    /// Content fingerprint per distinct `filePath`, read columnar from the
    /// live documents (like [`Self::count_documents_by_file_path`], no stored
    /// fields are touched).
    ///
    /// A value of 0 means "unverifiable": either the book was indexed without
    /// a fingerprint (e.g. PDF), or its live documents disagree (partial
    /// reindex) — callers should treat such books as changed or skip them.
    pub fn get_book_fingerprints(&self) -> Result<HashMap<String, u64>> {
        let searcher = self.index_reader.searcher();
        Ok(searcher.search(
            &AllQuery,
            &BookFingerprintCollector {
                hash_column: "contentHash",
            },
        )?)
    }

    /// Text-only fingerprint per distinct `filePath` — the `textHash` column
    /// ([`compute_content_fingerprint`] over the raw book text). Unlike
    /// [`Self::get_book_fingerprints`] (the canonical fingerprint, which also
    /// covers metadata such as catalogue order), this value shifts only when
    /// the book's text itself changed — the right signal for content-drift
    /// checks that must not be invalidated by adding/removing other books.
    ///
    /// A value of 0 means "unverifiable", same semantics as
    /// [`Self::get_book_fingerprints`].
    pub fn get_book_text_fingerprints(&self) -> Result<HashMap<String, u64>> {
        let searcher = self.index_reader.searcher();
        Ok(searcher.search(
            &AllQuery,
            &BookFingerprintCollector {
                hash_column: "textHash",
            },
        )?)
    }

    /// Text-only fingerprint of **one** book, by its `filePath` key — the
    /// single-book form of [`Self::get_book_text_fingerprints`]. A drift check
    /// for one open book must not pay O(total documents): this runs a
    /// `TermQuery` on `filePath` and reads `textHash` columnar from the
    /// matching live documents only.
    ///
    /// `0` means "unverifiable", exactly as in the map form: the book has no
    /// document in the index, was indexed without a text fingerprint (PDF), or
    /// its live documents disagree (a partial reindex).
    pub fn get_book_text_fingerprint(&self, file_path: String) -> Result<u64> {
        let file_path_f = self.schema.get_field("filePath")?;
        let query = TermQuery::new(
            Term::from_field_text(file_path_f, &file_path),
            IndexRecordOption::Basic,
        );
        let searcher = self.index_reader.searcher();
        Ok(searcher.search(&query, &SingleBookFingerprintCollector)?)
    }

    /// Fetch a single document by its numeric id. Returns None if not found.
    /// The `text` field contains the raw stored text (no snippet/highlight).
    pub fn get_document_by_id(&self, id: u64) -> Result<Option<SearchResult>> {
        let id_f = self.schema.get_field("id")?;
        let term = Term::from_field_u64(id_f, id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let searcher = self.index_reader.searcher();

        let top_docs = searcher.search(&query, &TopDocs::with_limit(1).order_by_score())?;
        let Some((_, addr)) = top_docs.into_iter().next() else {
            return Ok(None);
        };

        let doc = searcher.doc::<TantivyDocument>(addr)?;
        let title_f = self.schema.get_field("title")?;
        let reference_f = self.schema.get_field("reference")?;
        let text_f = self.schema.get_field("text")?;
        let segment_f = self.schema.get_field("segment")?;
        let is_pdf_f = self.schema.get_field("isPdf")?;
        let file_path_f = self.schema.get_field("filePath")?;

        Ok(Some(SearchResult {
            title: doc
                .get_first(title_f)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            reference: doc
                .get_first(reference_f)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            text: doc
                .get_first(text_f)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            id,
            segment: doc
                .get_first(segment_f)
                .and_then(|v| v.as_u64())
                .unwrap_or_default(),
            is_pdf: doc
                .get_first(is_pdf_f)
                .and_then(|v| v.as_bool())
                .unwrap_or_default(),
            file_path: doc
                .get_first(file_path_f)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            merged_count: 1,
            merged: Vec::new(),
        }))
    }

    /// Fuzzy (Levenshtein) search on pre-tokenized plain-text terms.
    /// Low-level primitive retained for tests and the example app; the
    /// high-level `search_fuzzy` accepts a raw query string instead.
    /// Multiple terms are ANDed together; each is matched within `max_distance`
    /// edits (0 = exact, 1–2 = fuzzy).
    pub fn search_fuzzy_terms(
        &self,
        terms: Vec<String>,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        max_distance: u8,
        order: ResultsOrder,
        highlight: Option<HighlightConfig>,
    ) -> Result<Vec<SearchResult>> {
        let rank = matches!(order, ResultsOrder::Relevance);
        let query = self.build_fuzzy_search_query(&terms, &facets, max_distance, rank)?;
        let hl = highlight.unwrap_or_else(HighlightConfig::default);
        self.run_search(
            query,
            |s| self.fuzzy_highlight_plan(s, &terms, max_distance),
            self.schema.get_field("text")?,
            limit,
            offset,
            &order,
            &hl,
            None,
        )
    }

    /// Stream search results in chunks of `chunk_size` documents.
    ///
    /// The TopDocs phase (scoring and ranking) completes upfront – this is
    /// inherent to how Tantivy's collectors work and cannot be avoided without
    /// a custom collector. What IS incremental is the stored-document retrieval
    /// and snippet generation: the Dart side receives the first chunk of results
    /// as soon as those are ready, without waiting for all snippets to be built.
    /// This is useful when `limit` is large and snippet generation is the
    /// bottleneck. For typical limits (≤ 200) the difference is negligible.
    ///
    /// Drops the single-word truncation flag — see [`Self::search`]; use
    /// [`Self::search_and_count`] ([`SearchPageResult::truncated`]) when
    /// partiality must surface.
    pub fn search_stream(
        &self,
        regex_terms: Vec<String>,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        slop: u32,
        max_expansions: u32,
        order: ResultsOrder,
        highlight: Option<HighlightConfig>,
        chunk_size: u32,
        sink: StreamSink<Vec<SearchResult>>,
    ) -> Result<()> {
        let result = (|| {
            let (query, _) = self.build_query(regex_terms, facets, slop, max_expansions)?;
            let hl = highlight.unwrap_or_else(HighlightConfig::default);
            self.run_search_stream(
                query,
                |_| Ok(HighlightPlan::none()),
                self.schema.get_field("text")?,
                limit,
                offset,
                &order,
                &hl,
                chunk_size,
                None,
                &sink,
            )
        })();
        Self::surface_stream_error(&sink, result)
    }

    // ── High-level mode-specific search API ──────────────────────────────────────
    //
    // These are the methods the otzaria app calls through its SearchEngineGateway.
    // Each builds the query for its mode (exact = Term/PhraseQuery, advanced =
    // morphological regex, fuzzy = FuzzyTermQuery) then routes through the shared
    // `run_*` executors. Snippet-returning methods apply the default `<font>`
    // highlight, which the app's snippet parser expects.

    // -- Exact -------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn search_exact(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        grouping: Option<ResultGrouping>,
    ) -> Result<Vec<SearchResult>> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, _) = self.build_exact_query(&query, &facets, &voc)?;
        self.run_search(
            q,
            |s| self.exact_highlight_plan(s, &query, &voc),
            self.search_text_field(&voc)?,
            limit,
            offset,
            &order,
            &HighlightConfig::default(),
            grouping.as_ref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_and_count_exact(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        grouping: Option<ResultGrouping>,
    ) -> Result<SearchPageResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, truncated) = self.build_exact_query(&query, &facets, &voc)?;
        self.run_search_and_count(
            q,
            |s| self.exact_highlight_plan(s, &query, &voc),
            self.search_text_field(&voc)?,
            limit,
            offset,
            &order,
            &HighlightConfig::default(),
            truncated,
            grouping.as_ref(),
        )
    }

    /// Collect scored, ungrouped Tantivy hits for the semantic coordinator.
    /// This bypasses snippet construction: the coordinator needs original line
    /// text and actual BM25 scores, while painting happens after fusion and
    /// pagination — see [`SemanticLexicalPhase::highlight`].
    #[cfg(feature = "semantic-integration")]
    fn semantic_exact_lexical_candidates(
        &self,
        query: &str,
        facets: &[String],
        limit: u32,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<SemanticLexicalPhase> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (search_query, truncated) = self.build_exact_query(query, facets, &voc)?;

        // Retrieval honours the vocalization flags, but painting is always
        // mark-free over the stored `text` field: that is the copy the sidecar
        // indexes and hydration reads back, so a vocalized highlight query would
        // expose no terms for it and leave every line unpainted. Facets are
        // dropped here — they filter documents, never highlights.
        let plain = VocalizedFlags::new(false, false);
        let (display_query, _) = self.build_exact_query(query, &[], &plain)?;
        let searcher = self.index_reader.searcher();
        let HighlightPlan {
            query: plan_query,
            phrase,
        } = Self::resolve_highlight(&searcher, |s| self.exact_highlight_plan(s, query, &plain));
        let highlight = SemanticHighlight {
            query: plan_query.unwrap_or(display_query),
            phrase,
        };
        self.semantic_candidates_from_query(search_query, limit, truncated, Some(highlight))
    }

    #[cfg(feature = "semantic-integration")]
    fn semantic_fuzzy_lexical_candidates(
        &self,
        query: &str,
        facets: &[String],
        limit: u32,
        max_distance: u8,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<SemanticLexicalPhase> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let token_texts = self.index_token_texts(query)?;
        let (search_query, truncated) = if voc.any() {
            self.build_fuzzy_query_vocalized(query, facets, max_distance, &voc)?
        } else {
            (
                self.build_fuzzy_search_query(&token_texts, facets, max_distance, true)?,
                false,
            )
        };

        // Mark-free for the same reason as the exact path. A fuzzy automaton
        // exposes no static terms, so without the materialized highlight query
        // there is nothing to paint with and the page falls back to bounded
        // plain snippets.
        let searcher = self.index_reader.searcher();
        let HighlightPlan {
            query: plan_query,
            phrase,
        } = Self::resolve_highlight(&searcher, |s| {
            self.fuzzy_highlight_plan(s, &token_texts, max_distance)
        });
        let highlight = plan_query.map(|query| SemanticHighlight { query, phrase });
        self.semantic_candidates_from_query(search_query, limit, truncated, highlight)
    }

    /// Build the painter for one page of sidecar results. Separate from
    /// candidate collection on purpose: creating the generator resolves the
    /// doc-frequency of every highlight term, so it happens once, after
    /// pagination, and only when there is a page to paint.
    #[cfg(feature = "semantic-integration")]
    fn semantic_snippet_painter(
        &self,
        highlight: SemanticHighlight,
    ) -> Result<SemanticSnippetPainter> {
        let searcher = self.index_reader.searcher();
        let hl = HighlightConfig::default();
        let generator = Self::make_snippet_generator(
            &searcher,
            highlight.query.as_ref(),
            self.schema.get_field("text")?,
            &hl,
        )?;
        Ok(SemanticSnippetPainter {
            searcher,
            generator,
            phrase: highlight.phrase,
            hl,
        })
    }

    #[cfg(feature = "semantic-integration")]
    fn semantic_candidates_from_query(
        &self,
        query: Box<dyn Query>,
        limit: u32,
        truncated: bool,
        highlight: Option<SemanticHighlight>,
    ) -> Result<SemanticLexicalPhase> {
        let searcher = self.index_reader.searcher();
        let collector = TopDocs::with_limit(limit as usize).order_by_score();
        let (hits, total_count): (Vec<(Score, DocAddress)>, usize) =
            searcher.search(&*query, &(collector, Count))?;

        let title_f = self.schema.get_field("title")?;
        let reference_f = self.schema.get_field("reference")?;
        let text_f = self.schema.get_field("text")?;
        let id_f = self.schema.get_field("id")?;
        let segment_f = self.schema.get_field("segment")?;
        let is_pdf_f = self.schema.get_field("isPdf")?;
        let file_path_f = self.schema.get_field("filePath")?;

        let mut candidates = Vec::with_capacity(hits.len());
        for (score, address) in hits {
            let document = searcher.doc::<TantivyDocument>(address)?;
            let reader = searcher.segment_reader(address.segment_ord);
            let fast = reader.fast_fields();
            let section_id = fast
                .u64("sectionId")?
                .first(address.doc_id)
                .unwrap_or_default();
            let line_hash = fast
                .u64("lineHash")?
                .first(address.doc_id)
                .unwrap_or_default();
            candidates.push(SidecarLexicalCandidate {
                title: document
                    .get_first(title_f)
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
                reference: document
                    .get_first(reference_f)
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
                text: document
                    .get_first(text_f)
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
                line_id: document
                    .get_first(id_f)
                    .and_then(|value| value.as_u64())
                    .unwrap_or_default(),
                section_id,
                line_hash,
                segment: document
                    .get_first(segment_f)
                    .and_then(|value| value.as_u64())
                    .unwrap_or_default(),
                is_pdf: document
                    .get_first(is_pdf_f)
                    .and_then(|value| value.as_bool())
                    .unwrap_or_default(),
                file_path: document
                    .get_first(file_path_f)
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
                bm25_score: score,
            });
        }
        Ok(SemanticLexicalPhase {
            candidates,
            total_count: total_count as u32,
            truncated,
            highlight,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_exact_stream(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        chunk_size: u32,
        grouping: Option<ResultGrouping>,
        sink: StreamSink<Vec<SearchResult>>,
    ) -> Result<()> {
        let result = (|| {
            let voc = VocalizedFlags::new(match_nikud, match_taamim);
            let (q, _) = self.build_exact_query(&query, &facets, &voc)?;
            self.run_search_stream(
                q,
                |s| self.exact_highlight_plan(s, &query, &voc),
                self.search_text_field(&voc)?,
                limit,
                offset,
                &order,
                &HighlightConfig::default(),
                chunk_size,
                grouping.as_ref(),
                &sink,
            )
        })();
        Self::surface_stream_error(&sink, result)
    }

    /// Like [`Self::search_exact_stream`] but the first event also carries
    /// the total hit count and per-book counts from the same index pass —
    /// replacing the separate `count_exact` + `count_by_book_exact` calls a
    /// search screen would otherwise issue for the same query.
    #[allow(clippy::too_many_arguments)]
    pub fn search_exact_stream_with_counts(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        chunk_size: u32,
        grouping: Option<ResultGrouping>,
        sink: StreamSink<SearchStreamUpdate>,
    ) -> Result<()> {
        let result = (|| {
            let voc = VocalizedFlags::new(match_nikud, match_taamim);
            // The mark-free exact path never degrades under a collection
            // budget; the vocalized single-word path can (its term set is
            // materialized like an advanced word).
            let (q, truncated) = self.build_exact_query(&query, &facets, &voc)?;
            self.run_search_stream_with_counts(
                q,
                |s| self.exact_highlight_plan(s, &query, &voc),
                self.search_text_field(&voc)?,
                limit,
                offset,
                &order,
                &HighlightConfig::default(),
                chunk_size,
                truncated,
                grouping.as_ref(),
                &sink,
            )
        })();
        Self::surface_stream_error(&sink, result)
    }

    /// Exact-mode hit count. Drops the truncation flag: the mark-free path
    /// never truncates, but the vocalized single-word path can (its term set
    /// is materialized like an advanced word) — use
    /// [`Self::count_exact_with_status`] when partiality must surface.
    pub fn count_exact(
        &self,
        query: String,
        facets: Vec<String>,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<u32> {
        Ok(self
            .count_exact_with_status(query, facets, match_nikud, match_taamim)?
            .count)
    }

    /// Like [`Self::count_exact`] but also reports whether the vocalized
    /// single-word term collection truncated.
    pub fn count_exact_with_status(
        &self,
        query: String,
        facets: Vec<String>,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<CountResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, truncated) = self.build_exact_query(&query, &facets, &voc)?;
        Ok(CountResult {
            count: self.run_count(q)?,
            truncated,
        })
    }

    /// Exact-mode per-book counts. Drops the truncation flag — see
    /// [`Self::count_exact`]; use [`Self::count_by_book_exact_with_status`].
    pub fn count_by_book_exact(
        &self,
        query: String,
        facets: Vec<String>,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<HashMap<String, u32>> {
        Ok(self
            .count_by_book_exact_with_status(query, facets, match_nikud, match_taamim)?
            .counts)
    }

    /// Like [`Self::count_by_book_exact`] but also reports whether the
    /// vocalized single-word term collection truncated.
    pub fn count_by_book_exact_with_status(
        &self,
        query: String,
        facets: Vec<String>,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<BookCountResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, truncated) = self.build_exact_query(&query, &facets, &voc)?;
        Ok(BookCountResult {
            counts: self.run_count_by_book(q)?,
            truncated,
        })
    }

    /// Exact-mode facet counts. Drops the truncation flag — see
    /// [`Self::count_exact`]; use [`Self::get_facet_counts_exact_with_status`].
    /// על שורשי הממדים תחת prefix `/` ראו [`Self::get_facet_counts`].
    pub fn get_facet_counts_exact(
        &self,
        query: String,
        facets: Vec<String>,
        facet_prefix: String,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<Vec<FacetCount>> {
        Ok(self
            .get_facet_counts_exact_with_status(
                query,
                facets,
                facet_prefix,
                match_nikud,
                match_taamim,
            )?
            .counts)
    }

    /// Like [`Self::get_facet_counts_exact`] but also reports whether the
    /// vocalized single-word term collection truncated.
    pub fn get_facet_counts_exact_with_status(
        &self,
        query: String,
        facets: Vec<String>,
        facet_prefix: String,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<FacetCountsResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, truncated) = self.build_exact_query(&query, &facets, &voc)?;
        Ok(FacetCountsResult {
            counts: self.run_facet_counts(q, facet_prefix)?,
            truncated,
        })
    }

    // -- Advanced ----------------------------------------------------------------

    pub fn search_advanced(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        grouping: Option<ResultGrouping>,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
    ) -> Result<Vec<SearchResult>> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let match_mode = WordMatch::from_api(word_match_mode, word_match_count);
        // מצב-שדה: הדגלים הגלובליים או אפשרות "ניקוד"/"טעמים" פר-מילה —
        // בחירת השדה וה-analyzer של ההדגשה, בעוד הדרישה פר-תו נגזרת פר-מילה
        // בתוך בניית השאילתה.
        let voc_mode = voc.or(hebrew_query::options_vocalized_flags(&search_options));
        let (q, regex_terms, gaps, truncated, acronym_alts) = self.build_advanced_query(
            &query,
            distance,
            &custom_spacing,
            &alternative_words,
            &search_options,
            facets,
            &voc,
            &scope,
            &match_mode,
        )?;
        let (q, _) = self.apply_advanced_negative_query(
            q,
            truncated,
            &negative_query,
            negative_distance,
            &negative_custom_spacing,
            &negative_alternative_words,
            &negative_search_options,
            &voc,
            &negative_scope,
        )?;
        self.run_search(
            q,
            |s| {
                self.advanced_highlight_plan_for_scope(
                    s,
                    &regex_terms,
                    &gaps,
                    &voc_mode,
                    &scope,
                    &match_mode,
                    &acronym_alts,
                )
            },
            self.search_text_field(&voc_mode)?,
            limit,
            offset,
            &order,
            &HighlightConfig::default(),
            grouping.as_ref(),
        )
    }

    pub fn search_and_count_advanced(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        grouping: Option<ResultGrouping>,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
    ) -> Result<SearchPageResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let match_mode = WordMatch::from_api(word_match_mode, word_match_count);
        let voc_mode = voc.or(hebrew_query::options_vocalized_flags(&search_options));
        let (q, regex_terms, gaps, truncated, acronym_alts) = self.build_advanced_query(
            &query,
            distance,
            &custom_spacing,
            &alternative_words,
            &search_options,
            facets,
            &voc,
            &scope,
            &match_mode,
        )?;
        let (q, truncated) = self.apply_advanced_negative_query(
            q,
            truncated,
            &negative_query,
            negative_distance,
            &negative_custom_spacing,
            &negative_alternative_words,
            &negative_search_options,
            &voc,
            &negative_scope,
        )?;
        self.run_search_and_count(
            q,
            |s| {
                self.advanced_highlight_plan_for_scope(
                    s,
                    &regex_terms,
                    &gaps,
                    &voc_mode,
                    &scope,
                    &match_mode,
                    &acronym_alts,
                )
            },
            self.search_text_field(&voc_mode)?,
            limit,
            offset,
            &order,
            &HighlightConfig::default(),
            truncated,
            grouping.as_ref(),
        )
    }

    /// Advanced-query result stream. Drops the single-word truncation flag —
    /// see [`Self::search`]; use [`Self::search_advanced_stream_with_counts`]
    /// (its first [`SearchStreamUpdate`] carries `truncated`) when the UI
    /// must flag partial results.
    pub fn search_advanced_stream(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        chunk_size: u32,
        grouping: Option<ResultGrouping>,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
        sink: StreamSink<Vec<SearchResult>>,
    ) -> Result<()> {
        let result = (|| {
            let voc = VocalizedFlags::new(match_nikud, match_taamim);
            let match_mode = WordMatch::from_api(word_match_mode, word_match_count);
            let voc_mode = voc.or(hebrew_query::options_vocalized_flags(&search_options));
            let (q, regex_terms, gaps, truncated, acronym_alts) = self.build_advanced_query(
                &query,
                distance,
                &custom_spacing,
                &alternative_words,
                &search_options,
                facets,
                &voc,
                &scope,
                &match_mode,
            )?;
            let (q, _) = self.apply_advanced_negative_query(
                q,
                truncated,
                &negative_query,
                negative_distance,
                &negative_custom_spacing,
                &negative_alternative_words,
                &negative_search_options,
                &voc,
                &negative_scope,
            )?;
            self.run_search_stream(
                q,
                |s| {
                    self.advanced_highlight_plan_for_scope(
                        s,
                        &regex_terms,
                        &gaps,
                        &voc_mode,
                        &scope,
                        &match_mode,
                        &acronym_alts,
                    )
                },
                self.search_text_field(&voc_mode)?,
                limit,
                offset,
                &order,
                &HighlightConfig::default(),
                chunk_size,
                grouping.as_ref(),
                &sink,
            )
        })();
        Self::surface_stream_error(&sink, result)
    }

    /// Like [`Self::search_advanced_stream`] but the first event also carries
    /// the total hit count and per-book counts from the same index pass —
    /// replacing the separate `count_advanced` + `count_by_book_advanced`
    /// calls a search screen would otherwise issue for the same query.
    #[allow(clippy::too_many_arguments)]
    pub fn search_advanced_stream_with_counts(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        chunk_size: u32,
        grouping: Option<ResultGrouping>,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
        sink: StreamSink<SearchStreamUpdate>,
    ) -> Result<()> {
        let result = (|| {
            let voc = VocalizedFlags::new(match_nikud, match_taamim);
            let match_mode = WordMatch::from_api(word_match_mode, word_match_count);
            let voc_mode = voc.or(hebrew_query::options_vocalized_flags(&search_options));
            let (q, regex_terms, gaps, truncated, acronym_alts) = self.build_advanced_query(
                &query,
                distance,
                &custom_spacing,
                &alternative_words,
                &search_options,
                facets,
                &voc,
                &scope,
                &match_mode,
            )?;
            let (q, truncated) = self.apply_advanced_negative_query(
                q,
                truncated,
                &negative_query,
                negative_distance,
                &negative_custom_spacing,
                &negative_alternative_words,
                &negative_search_options,
                &voc,
                &negative_scope,
            )?;
            self.run_search_stream_with_counts(
                q,
                |s| {
                    self.advanced_highlight_plan_for_scope(
                        s,
                        &regex_terms,
                        &gaps,
                        &voc_mode,
                        &scope,
                        &match_mode,
                        &acronym_alts,
                    )
                },
                self.search_text_field(&voc_mode)?,
                limit,
                offset,
                &order,
                &HighlightConfig::default(),
                chunk_size,
                truncated,
                grouping.as_ref(),
                &sink,
            )
        })();
        Self::surface_stream_error(&sink, result)
    }

    /// Advanced-query hit count. Drops the single-word truncation flag — see
    /// [`Self::count`]; use [`Self::count_advanced_with_status`] for UI.
    #[allow(clippy::too_many_arguments)]
    pub fn count_advanced(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
    ) -> Result<u32> {
        Ok(self
            .count_advanced_with_status(
                query,
                negative_query,
                facets,
                distance,
                negative_distance,
                custom_spacing,
                negative_custom_spacing,
                alternative_words,
                negative_alternative_words,
                search_options,
                negative_search_options,
                match_nikud,
                match_taamim,
                scope,
                negative_scope,
                word_match_mode,
                word_match_count,
            )?
            .count)
    }

    /// Like [`Self::count_advanced`] but also reports single-word truncation.
    #[allow(clippy::too_many_arguments)]
    pub fn count_advanced_with_status(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
    ) -> Result<CountResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, _, _, truncated, _) = self.build_advanced_query(
            &query,
            distance,
            &custom_spacing,
            &alternative_words,
            &search_options,
            facets,
            &voc,
            &scope,
            &WordMatch::from_api(word_match_mode, word_match_count),
        )?;
        let (q, truncated) = self.apply_advanced_negative_query(
            q,
            truncated,
            &negative_query,
            negative_distance,
            &negative_custom_spacing,
            &negative_alternative_words,
            &negative_search_options,
            &voc,
            &negative_scope,
        )?;
        Ok(CountResult {
            count: self.run_count(q)?,
            truncated,
        })
    }

    /// Advanced-query per-book counts. Drops the truncation flag — see
    /// [`Self::count`]; use [`Self::count_by_book_advanced_with_status`].
    #[allow(clippy::too_many_arguments)]
    pub fn count_by_book_advanced(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
    ) -> Result<HashMap<String, u32>> {
        Ok(self
            .count_by_book_advanced_with_status(
                query,
                negative_query,
                facets,
                distance,
                negative_distance,
                custom_spacing,
                negative_custom_spacing,
                alternative_words,
                negative_alternative_words,
                search_options,
                negative_search_options,
                match_nikud,
                match_taamim,
                scope,
                negative_scope,
                word_match_mode,
                word_match_count,
            )?
            .counts)
    }

    /// Like [`Self::count_by_book_advanced`] but also reports truncation.
    #[allow(clippy::too_many_arguments)]
    pub fn count_by_book_advanced_with_status(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
    ) -> Result<BookCountResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, _, _, truncated, _) = self.build_advanced_query(
            &query,
            distance,
            &custom_spacing,
            &alternative_words,
            &search_options,
            facets,
            &voc,
            &scope,
            &WordMatch::from_api(word_match_mode, word_match_count),
        )?;
        let (q, truncated) = self.apply_advanced_negative_query(
            q,
            truncated,
            &negative_query,
            negative_distance,
            &negative_custom_spacing,
            &negative_alternative_words,
            &negative_search_options,
            &voc,
            &negative_scope,
        )?;
        Ok(BookCountResult {
            counts: self.run_count_by_book(q)?,
            truncated,
        })
    }

    /// Advanced-query facet counts. Drops the truncation flag — see
    /// [`Self::count`]; use [`Self::get_facet_counts_advanced_with_status`].
    /// על שורשי הממדים תחת prefix `/` ראו [`Self::get_facet_counts`].
    #[allow(clippy::too_many_arguments)]
    pub fn get_facet_counts_advanced(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        facet_prefix: String,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
    ) -> Result<Vec<FacetCount>> {
        Ok(self
            .get_facet_counts_advanced_with_status(
                query,
                negative_query,
                facets,
                facet_prefix,
                distance,
                negative_distance,
                custom_spacing,
                negative_custom_spacing,
                alternative_words,
                negative_alternative_words,
                search_options,
                negative_search_options,
                match_nikud,
                match_taamim,
                scope,
                negative_scope,
                word_match_mode,
                word_match_count,
            )?
            .counts)
    }

    /// Like [`Self::get_facet_counts_advanced`] but also reports truncation.
    #[allow(clippy::too_many_arguments)]
    pub fn get_facet_counts_advanced_with_status(
        &self,
        query: String,
        negative_query: String,
        facets: Vec<String>,
        facet_prefix: String,
        distance: u32,
        negative_distance: u32,
        custom_spacing: HashMap<String, String>,
        negative_custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        negative_alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        negative_search_options: HashMap<String, HashMap<String, bool>>,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
        negative_scope: SearchScope,
        word_match_mode: Option<WordMatchMode>,
        word_match_count: Option<u32>,
    ) -> Result<FacetCountsResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, _, _, truncated, _) = self.build_advanced_query(
            &query,
            distance,
            &custom_spacing,
            &alternative_words,
            &search_options,
            facets,
            &voc,
            &scope,
            &WordMatch::from_api(word_match_mode, word_match_count),
        )?;
        let (q, truncated) = self.apply_advanced_negative_query(
            q,
            truncated,
            &negative_query,
            negative_distance,
            &negative_custom_spacing,
            &negative_alternative_words,
            &negative_search_options,
            &voc,
            &negative_scope,
        )?;
        Ok(FacetCountsResult {
            counts: self.run_facet_counts(q, facet_prefix)?,
            truncated,
        })
    }

    // -- Fuzzy -------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn search_fuzzy(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        max_distance: u8,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        grouping: Option<ResultGrouping>,
    ) -> Result<Vec<SearchResult>> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        if voc.any() {
            // The vocalized query is a materialized TermSetQuery per token —
            // it exposes its terms to the snippet generator by itself.
            let (q, _) = self.build_fuzzy_query_vocalized(&query, &facets, max_distance, &voc)?;
            return self.run_search(
                q,
                |_| Ok(HighlightPlan::none()),
                self.search_text_field(&voc)?,
                limit,
                offset,
                &order,
                &HighlightConfig::default(),
                grouping.as_ref(),
            );
        }
        let token_texts = self.index_token_texts(&query)?;
        let rank = matches!(order, ResultsOrder::Relevance);
        let q = self.build_fuzzy_search_query(&token_texts, &facets, max_distance, rank)?;
        self.run_search(
            q,
            |s| self.fuzzy_highlight_plan(s, &token_texts, max_distance),
            self.schema.get_field("text")?,
            limit,
            offset,
            &order,
            &HighlightConfig::default(),
            grouping.as_ref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_and_count_fuzzy(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        max_distance: u8,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        grouping: Option<ResultGrouping>,
    ) -> Result<SearchPageResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        if voc.any() {
            let (q, truncated) =
                self.build_fuzzy_query_vocalized(&query, &facets, max_distance, &voc)?;
            return self.run_search_and_count(
                q,
                |_| Ok(HighlightPlan::none()),
                self.search_text_field(&voc)?,
                limit,
                offset,
                &order,
                &HighlightConfig::default(),
                truncated,
                grouping.as_ref(),
            );
        }
        let token_texts = self.index_token_texts(&query)?;
        let rank = matches!(order, ResultsOrder::Relevance);
        let q = self.build_fuzzy_search_query(&token_texts, &facets, max_distance, rank)?;
        self.run_search_and_count(
            q,
            |s| self.fuzzy_highlight_plan(s, &token_texts, max_distance),
            self.schema.get_field("text")?,
            limit,
            offset,
            &order,
            &HighlightConfig::default(),
            false,
            grouping.as_ref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_fuzzy_stream(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        max_distance: u8,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        chunk_size: u32,
        grouping: Option<ResultGrouping>,
        sink: StreamSink<Vec<SearchResult>>,
    ) -> Result<()> {
        let result = (|| {
            let voc = VocalizedFlags::new(match_nikud, match_taamim);
            if voc.any() {
                let (q, _) =
                    self.build_fuzzy_query_vocalized(&query, &facets, max_distance, &voc)?;
                return self.run_search_stream(
                    q,
                    |_| Ok(HighlightPlan::none()),
                    self.search_text_field(&voc)?,
                    limit,
                    offset,
                    &order,
                    &HighlightConfig::default(),
                    chunk_size,
                    grouping.as_ref(),
                    &sink,
                );
            }
            let token_texts = self.index_token_texts(&query)?;
            let rank = matches!(order, ResultsOrder::Relevance);
            let q = self.build_fuzzy_search_query(&token_texts, &facets, max_distance, rank)?;
            self.run_search_stream(
                q,
                |s| self.fuzzy_highlight_plan(s, &token_texts, max_distance),
                self.schema.get_field("text")?,
                limit,
                offset,
                &order,
                &HighlightConfig::default(),
                chunk_size,
                grouping.as_ref(),
                &sink,
            )
        })();
        Self::surface_stream_error(&sink, result)
    }

    /// Like [`Self::search_fuzzy_stream`] but the first event also carries
    /// the total hit count and per-book counts from the same index pass —
    /// replacing the separate `count_fuzzy` + `count_by_book_fuzzy` calls a
    /// search screen would otherwise issue for the same query.
    #[allow(clippy::too_many_arguments)]
    pub fn search_fuzzy_stream_with_counts(
        &self,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        max_distance: u8,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        chunk_size: u32,
        grouping: Option<ResultGrouping>,
        sink: StreamSink<SearchStreamUpdate>,
    ) -> Result<()> {
        let result = (|| {
            let voc = VocalizedFlags::new(match_nikud, match_taamim);
            if voc.any() {
                let (q, truncated) =
                    self.build_fuzzy_query_vocalized(&query, &facets, max_distance, &voc)?;
                return self.run_search_stream_with_counts(
                    q,
                    |_| Ok(HighlightPlan::none()),
                    self.search_text_field(&voc)?,
                    limit,
                    offset,
                    &order,
                    &HighlightConfig::default(),
                    chunk_size,
                    truncated,
                    grouping.as_ref(),
                    &sink,
                );
            }
            let token_texts = self.index_token_texts(&query)?;
            let rank = matches!(order, ResultsOrder::Relevance);
            let q = self.build_fuzzy_search_query(&token_texts, &facets, max_distance, rank)?;
            self.run_search_stream_with_counts(
                q,
                |s| self.fuzzy_highlight_plan(s, &token_texts, max_distance),
                self.schema.get_field("text")?,
                limit,
                offset,
                &order,
                &HighlightConfig::default(),
                chunk_size,
                // The fuzzy path uses its own automaton budgets, not the
                // single-word degrade mechanism.
                false,
                grouping.as_ref(),
                &sink,
            )
        })();
        Self::surface_stream_error(&sink, result)
    }

    /// Builds the fuzzy-mode query for the count family: the vocalized path
    /// carries its truncation flag; the mark-free path uses its own automaton
    /// budgets, not the single-word degrade mechanism, so it never truncates.
    fn build_fuzzy_count_query(
        &self,
        query: &str,
        facets: &[String],
        max_distance: u8,
        voc: &VocalizedFlags,
    ) -> Result<(Box<dyn Query>, bool)> {
        if voc.any() {
            self.build_fuzzy_query_vocalized(query, facets, max_distance, voc)
        } else {
            Ok((self.build_fuzzy_query(query, facets, max_distance)?, false))
        }
    }

    /// Fuzzy-mode hit count. Drops the truncation flag: the mark-free path
    /// never truncates, but the vocalized path can (its term set is
    /// materialized like an advanced word) — use
    /// [`Self::count_fuzzy_with_status`] when partiality must surface.
    pub fn count_fuzzy(
        &self,
        query: String,
        facets: Vec<String>,
        max_distance: u8,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<u32> {
        Ok(self
            .count_fuzzy_with_status(query, facets, max_distance, match_nikud, match_taamim)?
            .count)
    }

    /// Like [`Self::count_fuzzy`] but also reports whether the vocalized
    /// term collection truncated.
    pub fn count_fuzzy_with_status(
        &self,
        query: String,
        facets: Vec<String>,
        max_distance: u8,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<CountResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, truncated) = self.build_fuzzy_count_query(&query, &facets, max_distance, &voc)?;
        Ok(CountResult {
            count: self.run_count(q)?,
            truncated,
        })
    }

    /// Fuzzy-mode per-book counts. Drops the truncation flag — see
    /// [`Self::count_fuzzy`]; use [`Self::count_by_book_fuzzy_with_status`].
    pub fn count_by_book_fuzzy(
        &self,
        query: String,
        facets: Vec<String>,
        max_distance: u8,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<HashMap<String, u32>> {
        Ok(self
            .count_by_book_fuzzy_with_status(
                query,
                facets,
                max_distance,
                match_nikud,
                match_taamim,
            )?
            .counts)
    }

    /// Like [`Self::count_by_book_fuzzy`] but also reports whether the
    /// vocalized term collection truncated.
    pub fn count_by_book_fuzzy_with_status(
        &self,
        query: String,
        facets: Vec<String>,
        max_distance: u8,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<BookCountResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, truncated) = self.build_fuzzy_count_query(&query, &facets, max_distance, &voc)?;
        Ok(BookCountResult {
            counts: self.run_count_by_book(q)?,
            truncated,
        })
    }

    /// Fuzzy-mode facet counts. Drops the truncation flag — see
    /// [`Self::count_fuzzy`]; use [`Self::get_facet_counts_fuzzy_with_status`].
    /// על שורשי הממדים תחת prefix `/` ראו [`Self::get_facet_counts`].
    pub fn get_facet_counts_fuzzy(
        &self,
        query: String,
        facets: Vec<String>,
        facet_prefix: String,
        max_distance: u8,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<Vec<FacetCount>> {
        Ok(self
            .get_facet_counts_fuzzy_with_status(
                query,
                facets,
                facet_prefix,
                max_distance,
                match_nikud,
                match_taamim,
            )?
            .counts)
    }

    /// Like [`Self::get_facet_counts_fuzzy`] but also reports whether the
    /// vocalized term collection truncated.
    pub fn get_facet_counts_fuzzy_with_status(
        &self,
        query: String,
        facets: Vec<String>,
        facet_prefix: String,
        max_distance: u8,
        match_nikud: bool,
        match_taamim: bool,
    ) -> Result<FacetCountsResult> {
        let voc = VocalizedFlags::new(match_nikud, match_taamim);
        let (q, truncated) = self.build_fuzzy_count_query(&query, &facets, max_distance, &voc)?;
        Ok(FacetCountsResult {
            counts: self.run_facet_counts(q, facet_prefix)?,
            truncated,
        })
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn all_fields(&self) -> Result<SchemaFields> {
        Ok((
            self.schema.get_field("title")?,
            self.schema.get_field("reference")?,
            self.schema.get_field("text")?,
            self.schema.get_field("id")?,
            self.schema.get_field("segment")?,
            self.schema.get_field("isPdf")?,
            self.schema.get_field("filePath")?,
            self.schema.get_field("topics")?,
            self.schema.get_field("contentHash")?,
            self.schema.get_field("textHash")?,
            self.schema.get_field("textVocalized")?,
            self.schema.get_field("sectionId")?,
            self.schema.get_field("generationSort")?,
            self.schema.get_field("lineHash")?,
        ))
    }

    fn ensure_writer(&mut self) -> Result<()> {
        if self.index_writer.is_none() {
            debug!("writer: reopening lazily");
            self.index_writer = Some(self.open_writer()?);
        }
        Ok(())
    }

    fn writer_mut(&mut self) -> Result<&mut IndexWriter> {
        self.ensure_writer()?;
        self.index_writer
            .as_mut()
            .context("index writer is not available")
    }

    fn take_writer(&mut self) -> Result<IndexWriter> {
        self.ensure_writer()?;
        self.index_writer
            .take()
            .context("index writer is not available")
    }

    fn open_writer(&self) -> Result<IndexWriter> {
        let writer = self.index.writer(self.writer_heap_size)?;
        if self.bulk_indexing {
            writer.set_merge_policy(Box::new(NoMergePolicy));
        }
        Ok(writer)
    }

    fn open_writer_no_merge(&self) -> Result<IndexWriter> {
        let writer = self.open_writer()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        Ok(writer)
    }

    fn optimize_committed_segments(&self) -> Result<()> {
        let metas = self.index.searchable_segment_metas()?;
        let Some(merge_ids) = select_segments_to_compact(&metas) else {
            debug!("optimize: {} segments, nothing to compact", metas.len());
            return Ok(());
        };
        debug!(
            "optimize: merging {} of {} segments",
            merge_ids.len(),
            metas.len()
        );

        let mut maintenance_writer = self.open_writer_no_merge()?;
        let merge_result = maintenance_writer.merge(&merge_ids).wait().map(|_| ());
        let wait_result = maintenance_writer.wait_merging_threads();

        merge_result?;
        wait_result?;
        Ok(())
    }

    fn restore_writer(&mut self) -> Result<()> {
        self.index_writer = Some(self.open_writer()?);
        Ok(())
    }

    /// String-API entry: terms arriving as raw regex strings (the public
    /// `search`/`count` family) are split on their top-level alternation so a
    /// single-word query compiles per branch, exactly like the advanced path.
    /// `slop` here means what it means everywhere in this engine: the
    /// intermediate-word allowance between *each* adjacent pair (uniform, in
    /// order) — not tantivy's cumulative unordered budget.
    fn build_query(
        &self,
        regex_terms: Vec<String>,
        facets: Vec<String>,
        slop: u32,
        max_expansions: u32,
    ) -> Result<(Box<dyn Query>, bool)> {
        let patterns: Vec<hebrew_query::WordPattern> = regex_terms
            .iter()
            .map(|t| hebrew_query::WordPattern::parse(t))
            .collect();
        let gaps = vec![slop; patterns.len().saturating_sub(1)];
        let text_field = self.schema.get_field("text")?;
        self.build_query_from_patterns(
            patterns,
            &[],
            &[],
            facets,
            &gaps,
            max_expansions,
            text_field,
            &SearchScope::WordDistance,
            &WordMatch::All,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_advanced_negative_query(
        &self,
        positive_query: Box<dyn Query>,
        positive_truncated: bool,
        negative_query: &str,
        negative_distance: u32,
        negative_custom_spacing: &HashMap<String, String>,
        negative_alternative_words: &HashMap<u32, Vec<String>>,
        negative_search_options: &HashMap<String, HashMap<String, bool>>,
        voc: &VocalizedFlags,
        negative_scope: &SearchScope,
    ) -> Result<(Box<dyn Query>, bool)> {
        if hebrew_query::split_query_words(negative_query).is_empty() {
            return Ok((positive_query, positive_truncated));
        }

        // שלילה בהתאמה חלקית הייתה פוסלת כל תוצאה שנושאת ולו מילת-שלילה
        // אחת — לכן צירוף מילות השלילה נשאר תמיד "כל המילים".
        let (negative, _, _, negative_truncated, _) = self.build_advanced_query(
            negative_query,
            negative_distance,
            negative_custom_spacing,
            negative_alternative_words,
            negative_search_options,
            Vec::new(),
            voc,
            negative_scope,
            &WordMatch::All,
        )?;

        // שלילה בטווח "תחת אותה כותרת" פוסלת *סעיפים שלמים*: השאילתה שנבנתה
        // מחזירה רק את השורות שנושאות מילת שלילה, כך שב-MustNot היא הייתה
        // מותירה את שאר שורות הסעיף. במקומה נאספים ה-sectionId שהיא חותכת
        // וכל שורה בסעיף כזה נחסמת.
        let negative: Box<dyn Query> = if matches!(negative_scope, SearchScope::SameSection) {
            let sections = self
                .index_reader
                .searcher()
                .search(negative.as_ref(), &SectionIdsCollector)?;
            if sections.is_empty() {
                return Ok((positive_query, positive_truncated || negative_truncated));
            }
            Box::new(SectionFilteredQuery::new(
                Box::new(AllQuery),
                Arc::new(sections),
            ))
        } else {
            negative
        };

        Ok((
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, positive_query),
                (Occur::MustNot, negative),
            ])),
            positive_truncated || negative_truncated,
        ))
    }

    /// `text_field` selects the dictionary the patterns run against: the
    /// plain `text` field, or `textVocalized` on the vocalized paths (whose
    /// patterns carry free-mark runs and must never hit the plain field).
    ///
    /// `gaps` is the per-pair intermediate-word allowance (`gaps[i]` between
    /// words `i` and `i+1`). The phrase branch goes through
    /// [`Self::phrase_query_with_degrade`]: on its exact path tantivy gets
    /// the *sum* as its slop — tantivy spends slop cumulatively across the
    /// phrase, so the max would reject a match using its allowance in two
    /// different gaps — wrapped in [`GapVerifiedPhraseQuery`], which
    /// re-checks candidates against the positional postings so only
    /// in-order, per-pair-within-allowance occurrences survive; on its
    /// degrade path [`TermListPhraseQuery`] applies the same per-pair
    /// verification directly.
    #[allow(clippy::too_many_arguments)]
    fn build_query_from_patterns(
        &self,
        regex_terms: Vec<hebrew_query::WordPattern>,
        source_words: &[String],
        typo_tokens: &[String],
        facets: Vec<String>,
        gaps: &[u32],
        max_expansions: u32,
        text_field: Field,
        scope: &SearchScope,
        match_mode: &WordMatch,
    ) -> Result<(Box<dyn Query>, bool)> {
        // Resolved up front: the same facet filter both narrows the section
        // pre-pass (fewer candidate sections to intersect) and gates the
        // final result set. סמנטיקת הממדים (OR בתוך ממד, AND ביניהם)
        // נבנית ב-facet_filter_query.
        let facets_query: Option<Box<dyn Query>> = if facets.is_empty() {
            None
        } else {
            Some(self.facet_filter_query(&facets)?)
        };

        // Every word-materialization path (single word / paragraph / section
        // scopes / the phrase degrade path) degrades under its collection
        // budgets and reports truncation; only the empty branch matches
        // nothing and reports none.
        let (main_query, truncated): (Box<dyn Query>, bool) = match regex_terms.len() {
            0 => (Box::new(EmptyQuery), false),
            1 => self.single_regex_term_query(
                regex_terms[0].branches(),
                typo_tokens,
                text_field,
                max_expansions,
            )?,
            _ => match (scope, match_mode) {
                // הטווחים "פסקה"/"כותרת" מוותרים על סדר ומרווח, וכך גם כל
                // מצב התאמה שאינו "כל המילים" (מילים עשויות לחסור) — כל
                // מילה מתממשת ל-TermSetQuery משלה והצירוף נעשה בין מסמכים
                // (פסקה) או בין סעיפים (כותרת).
                (SearchScope::SameParagraph | SearchScope::SameSection, _)
                | (_, WordMatch::AtLeast(_) | WordMatch::Most) => self.scoped_words_query(
                    &regex_terms,
                    source_words,
                    text_field,
                    max_expansions,
                    scope,
                    facets_query.as_deref(),
                    match_mode,
                )?,
                (SearchScope::WordDistance, WordMatch::All) => {
                    debug_assert_eq!(gaps.len() + 1, regex_terms.len());
                    self.phrase_query_with_degrade(&regex_terms, gaps, text_field, max_expansions)?
                }
            },
        };

        let Some(facets_query) = facets_query else {
            return Ok((main_query, truncated));
        };

        Ok((
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, main_query),
                (Occur::Must, facets_query),
            ])),
            truncated,
        ))
    }

    /// The ordered-phrase path (`WordDistance` + all words), with graceful
    /// degradation instead of tantivy's hard expansion error.
    ///
    /// The joined patterns are first counted with *Tantivy's own semantics*:
    /// one cumulative expansion count per segment. Only a phrase that would
    /// exceed that limit (or whose joined DFA cannot compile) materializes
    /// per-position term sets for the fallback engine. This keeps the normal
    /// `RegexPhraseQuery` path exact — including its BM25 phrase scoring —
    /// and avoids turning a large multi-segment union into a false fallback.
    ///
    /// - **Exact path** — every segment's cumulative count fits
    ///   `max_expansions`, and every joined pattern compiles as one DFA: the
    ///   historical `RegexPhraseQuery` (+ [`GapVerifiedPhraseQuery`] when
    ///   slop > 0), with identical scoring and behavior.
    /// - **Degrade path** — otherwise: a [`TermListPhraseQuery`] built from
    ///   the materialized (possibly truncated) term sets. No joined DFA, no
    ///   expansion ceiling to overflow — a phrase that used to die with
    ///   tantivy's `InvalidArgument("Phrase query exceeded max expansions")`
    ///   now serves complete results when only the *cumulative* ceiling was
    ///   the problem, and budget-truncated (highest-priority-first) results
    ///   with `truncated: true` when a single position outgrew its own caps.
    fn phrase_query_with_degrade(
        &self,
        regex_terms: &[hebrew_query::WordPattern],
        gaps: &[u32],
        text_field: Field,
        max_expansions: u32,
    ) -> Result<(Box<dyn Query>, bool)> {
        let joined: Vec<String> = regex_terms
            .iter()
            .map(hebrew_query::WordPattern::joined)
            .collect();
        let joined_compiles = joined.iter().all(|p| tantivy_fst::Regex::new(p).is_ok());
        if joined_compiles
            && !self.phrase_exceeds_max_expansions(&joined, text_field, max_expansions)?
        {
            let slop_budget = gaps.iter().fold(0u32, |acc, &g| acc.saturating_add(g));
            let mut phrase_query = RegexPhraseQuery::new(text_field, joined.clone());
            phrase_query.set_slop(slop_budget);
            phrase_query.set_max_expansions(max_expansions);
            if slop_budget == 0 {
                return Ok((Box::new(phrase_query), false));
            }
            return Ok((
                Box::new(GapVerifiedPhraseQuery::new(
                    phrase_query,
                    text_field,
                    joined,
                    gaps.to_vec(),
                )),
                false,
            ));
        }

        if !joined_compiles {
            info!(
                "joined phrase pattern exceeds the DFA limits; serving the term-list degrade path"
            );
        } else {
            info!(
                "phrase expansions exceed the exact RegexPhraseQuery ceiling ({max_expansions}) \
                 in at least one segment; serving the term-list degrade path"
            );
        }

        // Materialize every *distinct* branch set once, in parallel, only
        // after the exact engine is known to be unusable. This is the costly
        // work the fallback actually needs; doing it for all phrases both
        // duplicated Tantivy's normal FST scan and changed valid phrases'
        // ranking on multi-segment indexes.
        use rayon::prelude::*;
        let mut unique_branches: Vec<&[String]> = Vec::new();
        let mut set_index: Vec<usize> = Vec::with_capacity(regex_terms.len());
        for pattern in regex_terms {
            let branches = pattern.branches();
            match unique_branches.iter().position(|u| *u == branches) {
                Some(i) => set_index.push(i),
                None => {
                    set_index.push(unique_branches.len());
                    unique_branches.push(branches);
                }
            }
        }
        let unique_sets: Vec<CachedTermSet> = unique_branches
            .par_iter()
            .copied()
            .map(|branches| {
                self.materialize_term_set(
                    branches,
                    &[],
                    text_field,
                    PHRASE_POSITION_MAX_EXPANSIONS,
                    PHRASE_POSITION_POSTINGS_BUDGET,
                )
            })
            .collect::<Result<_>>()?;

        let mut truncated = false;
        let mut position_terms: Vec<Vec<Term>> = Vec::with_capacity(regex_terms.len());
        for &set_i in &set_index {
            let entry = &unique_sets[set_i];
            if entry.terms.is_empty() {
                return Ok((Box::new(EmptyQuery), false));
            }
            truncated |= entry.truncated;
            position_terms.push(entry.terms.as_ref().clone());
        }
        Ok((
            Box::new(TermListPhraseQuery::new(
                text_field,
                position_terms,
                gaps.to_vec(),
            )),
            truncated,
        ))
    }

    /// Mirrors `RegexPhraseWeight`'s expansion accounting exactly: the
    /// matching terms of every word position are counted cumulatively, but
    /// the counter resets for each segment. A global union is not equivalent
    /// here — disjoint vocabularies in several segments may be larger than
    /// `max_expansions` together while every segment remains valid for
    /// Tantivy's BM25 phrase scorer.
    fn phrase_exceeds_max_expansions(
        &self,
        joined_patterns: &[String],
        text_field: Field,
        max_expansions: u32,
    ) -> Result<bool> {
        let mut regexes = Vec::with_capacity(joined_patterns.len());
        for pattern in joined_patterns {
            regexes.push(tantivy_fst::Regex::new(pattern)?);
        }
        let searcher = self.index_reader.searcher();
        for reader in searcher.segment_readers() {
            let inverted = reader.inverted_index(text_field)?;
            let mut segment_terms = 0usize;
            for regex in &regexes {
                let mut stream = inverted.terms().search(regex).into_stream()?;
                while stream.advance() {
                    segment_terms += 1;
                    if segment_terms > max_expansions as usize {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Multi-word query for the paragraph/section scopes (and every partial
    /// match mode): every distinct word is materialized into its own
    /// `TermSetQuery` (same budgets and cache as the single-word path —
    /// each word executes exactly like it would alone), then combined:
    ///
    /// - **Paragraph** (וגם `WordDistance` בהתאמה חלקית) — boolean בין
    ///   מסמכים: כשכל המילים נדרשות — AND; פחות מזה — Should עם מינימום
    ///   נדרש (מסמך עם יותר מילים מקבל score גבוה יותר).
    /// - **Section** — a two-pass plan (see the `section_scope` module):
    ///   keep the sections whose per-word `sectionId` sets cover at least
    ///   the required word count (intersection when all are required),
    ///   then serve the lines that carry any query word inside such a
    ///   section. The facet filter narrows the pre-pass so a facet-excluded
    ///   book cannot bloat the section sets (correctness never depends on
    ///   it — section ids are unique per book).
    ///
    /// הסף נגזר מ-`match_mode` על מספר המילים **הייחודיות**: מופעים כפולים
    /// של אותה מילת-מקור מתמזגים ל-clause אחד (איחוד הענפים של כולם — גם
    /// כשאפשרויות פר-מילה נותנות להם תבניות שונות), אחרת clause כפול היה
    /// מספק את הסף פעמיים. בהיעדר מילות מקור (ה-API המחרוזתי, חלופות ר"ת)
    /// התבנית עצמה היא מפתח המיזוג.
    ///
    /// `truncated` is the OR over the per-word collection truncations.
    #[allow(clippy::too_many_arguments)]
    fn scoped_words_query(
        &self,
        regex_terms: &[hebrew_query::WordPattern],
        source_words: &[String],
        text_field: Field,
        max_expansions: u32,
        scope: &SearchScope,
        facets_query: Option<&dyn Query>,
        match_mode: &WordMatch,
    ) -> Result<(Box<dyn Query>, bool)> {
        let mut by_word: HashMap<String, usize> = HashMap::with_capacity(regex_terms.len());
        let mut merged_branches: Vec<Vec<String>> = Vec::with_capacity(regex_terms.len());
        for (i, pattern) in regex_terms.iter().enumerate() {
            let key = source_words
                .get(i)
                .cloned()
                .unwrap_or_else(|| pattern.joined());
            match by_word.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    let branches = &mut merged_branches[*e.get()];
                    for branch in pattern.branches() {
                        if !branches.iter().any(|b| b == branch) {
                            branches.push(branch.clone());
                        }
                    }
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(merged_branches.len());
                    merged_branches.push(pattern.branches().to_vec());
                }
            }
        }
        // Every distinct word materializes independently (same scans, same
        // cache) — fan them out over the pool like the phrase path does.
        use rayon::prelude::*;
        let materialized: Vec<(Box<dyn Query>, bool)> = merged_branches
            .par_iter()
            .map(|branches| self.single_regex_term_query(branches, &[], text_field, max_expansions))
            .collect::<Result<_>>()?;
        let mut truncated = false;
        let mut word_queries: Vec<Box<dyn Query>> = Vec::with_capacity(materialized.len());
        for (word_query, word_truncated) in materialized {
            truncated |= word_truncated;
            word_queries.push(word_query);
        }
        let min_required = match_mode.min_words(word_queries.len());
        let all_required = min_required >= word_queries.len();

        // סף של מילה אחת שקול ל-OR שטוח גם בטווח הסעיף: שורה שנושאת מילה
        // כלשהי שייכת ממילא לסעיף שמכיל אותה — ה-pre-pass מיותר.
        if !matches!(scope, SearchScope::SameSection) || min_required <= 1 {
            let query: Box<dyn Query> = if all_required {
                let clauses: Vec<(Occur, Box<dyn Query>)> =
                    word_queries.into_iter().map(|q| (Occur::Must, q)).collect();
                Box::new(BooleanQuery::new(clauses))
            } else {
                let clauses: Vec<(Occur, Box<dyn Query>)> = word_queries
                    .into_iter()
                    .map(|q| (Occur::Should, q))
                    .collect();
                Box::new(BooleanQuery::with_minimum_required_clauses(
                    clauses,
                    min_required,
                ))
            };
            return Ok((query, truncated));
        }

        // Section scope — pass 1: the sections carrying enough of the words.
        let searcher = self.index_reader.searcher();
        let mut allowed: Option<HashSet<u64>> = None;
        let mut section_word_counts: HashMap<u64, usize> = HashMap::new();
        for word_query in &word_queries {
            let sections = match facets_query {
                Some(fq) => searcher.search(
                    &BooleanQuery::new(vec![
                        (Occur::Must, word_query.box_clone()),
                        (Occur::Must, fq.box_clone()),
                    ]),
                    &SectionIdsCollector,
                )?,
                None => searcher.search(word_query.as_ref(), &SectionIdsCollector)?,
            };
            if !all_required {
                truncated |= accumulate_section_counts(
                    &mut section_word_counts,
                    sections,
                    SECTION_COUNT_BUDGET,
                );
                continue;
            }
            allowed = Some(match allowed {
                None => sections,
                Some(prev) => prev.intersection(&sections).copied().collect(),
            });
            if allowed.as_ref().is_some_and(HashSet::is_empty) {
                // A word with no section in common — no section can ever
                // contain all the words.
                return Ok((Box::new(EmptyQuery), truncated));
            }
        }
        if !all_required {
            let reaching: HashSet<u64> = section_word_counts
                .into_iter()
                .filter(|(_, count)| *count >= min_required)
                .map(|(section, _)| section)
                .collect();
            if reaching.is_empty() {
                return Ok((Box::new(EmptyQuery), truncated));
            }
            allowed = Some(reaching);
        }

        // Pass 2: the lines that carry any query word, gated to the
        // intersected sections.
        let union = BooleanQuery::new(
            word_queries
                .into_iter()
                .map(|q| (Occur::Should, q))
                .collect::<Vec<_>>(),
        );
        Ok((
            Box::new(SectionFilteredQuery::new(
                Box::new(union),
                Arc::new(allowed.unwrap_or_default()),
            )),
            truncated,
        ))
    }

    /// Single regex term: materialize the matching index terms into a
    /// `TermSetQuery`, bounded by two budgets — `max_expansions` (term count,
    /// a memory guard on the materialized `Vec<Term>`) and
    /// [`SINGLE_WORD_POSTINGS_BUDGET`] (summed doc_freq, the real execution
    /// cost). Unlike `RegexPhraseQuery`, overflow here *degrades*: collection
    /// stops and the highest-priority automatons collected so far are served
    /// (never an error). A bare `RegexQuery` would enumerate the term
    /// dictionary without any bound, so a broad pattern (e.g. a 1-char word
    /// with prefix+suffix options) could scan a huge slice of the index
    /// unchecked.
    ///
    /// Each alternation branch is compiled as its own DFA: a combined
    /// `(?:b1|…|bN)` of wildcard-wrapped branches overlaps so heavily that it
    /// blew the upstream tantivy-fst 1 000-state cap (48 typo+partial
    /// branches — 806 chars; the vendored cap is 8 192) while every branch
    /// alone is tiny. `typo_tokens` add one Levenshtein-1 automaton each,
    /// scanned after all branches. Everything streams into one shared
    /// `HashSet` under the shared budgets, so the resulting `TermSetQuery`
    /// is exactly the union of what every automaton matches.
    fn single_regex_term_query(
        &self,
        branches: &[String],
        typo_tokens: &[String],
        text_field: Field,
        max_expansions: u32,
    ) -> Result<(Box<dyn Query>, bool)> {
        let entry = self.materialize_term_set(
            branches,
            typo_tokens,
            text_field,
            max_expansions,
            SINGLE_WORD_POSTINGS_BUDGET,
        )?;
        Ok((
            Box::new(TermSetQuery::new(entry.terms.iter().cloned())),
            entry.truncated,
        ))
    }

    /// The materialization behind [`Self::single_regex_term_query`] and the
    /// phrase degrade path: streams every branch's (and typo automaton's)
    /// dictionary matches into one term set under the two collection budgets,
    /// served from / stored into the LRU term cache. Overflow degrades —
    /// collection stops at the budget and the highest-priority prefix is
    /// returned with `truncated: true`, never an error.
    fn materialize_term_set(
        &self,
        branches: &[String],
        typo_tokens: &[String],
        text_field: Field,
        max_expansions: u32,
        postings_budget: u64,
    ) -> Result<CachedTermSet> {
        let searcher = self.index_reader.searcher();
        // The materialization below (per-branch DFA compile + one FST scan
        // per branch per segment) is the expensive part of a search, and one
        // user search repeats it verbatim across stream/count/count-by-book/
        // facet/pagination calls — serve those from the cache. The searcher
        // generation in the key invalidates entries on reader reload.
        let cache_key = TermCacheKey {
            generation: searcher.generation().generation_id(),
            field: text_field,
            branches: branches.to_vec(),
            typo_tokens: typo_tokens.to_vec(),
            max_expansions,
            postings_budget,
        };
        if let Some(entry) = self.term_cache.lock().unwrap().get(&cache_key) {
            return Ok(entry.clone());
        }

        let regexes: Vec<tantivy_fst::Regex> = branches
            .iter()
            .map(|branch| {
                tantivy_fst::Regex::new(branch).map_err(|e| {
                    // Surface the failing branch loudly: the historical
                    // failure mode here was a compile error silently becoming
                    // "0 results" in the UI.
                    error!(
                        "regex branch compilation failed ({} chars): {e}. Branch prefix: {}",
                        branch.chars().count(),
                        branch.chars().take(80).collect::<String>(),
                    );
                    anyhow::anyhow!(
                        "invalid regex branch ({} chars): {e}",
                        branch.chars().count()
                    )
                })
            })
            .collect::<Result<_>>()?;
        let inverted_indexes = searcher
            .segment_readers()
            .iter()
            .map(|reader| reader.inverted_index(text_field))
            .collect::<tantivy::Result<Vec<_>>>()?;
        let mut matched: HashSet<String> = HashSet::new();
        // Sum of doc_freq over every stream hit — the real cost of executing
        // the TermSetQuery (BitSet union of one postings list per matched term
        // per segment). A term matched by several automatons in the same
        // segment is counted once per automaton — a slight over-estimate that
        // only errs toward earlier truncation.
        let mut postings_cost: u64 = 0;
        let mut truncated = false;
        // Automatons run most-important-first: branches (exact forms before
        // typo variants — the `build_word_regex` contract), then the
        // Levenshtein typo automatons. Each automaton covers *all* segments
        // before the next starts, so when a budget runs out mid-collection
        // the query degrades to the highest-priority automaton prefix rather
        // than over-serving whichever segment happened to be scanned first.
        'branches: for regex in &regexes {
            for inverted in &inverted_indexes {
                if Self::collect_automaton_terms(
                    inverted,
                    regex,
                    max_expansions,
                    postings_budget,
                    &mut matched,
                    &mut postings_cost,
                )? {
                    truncated = true;
                    break 'branches;
                }
            }
        }
        if !truncated && !typo_tokens.is_empty() {
            // Same builder configuration as the fuzzy path (distance 1,
            // transposition counts as one edit): the whole edit-distance-1
            // neighborhood in one scan per token per segment, replacing the
            // ≤128 sampled literal-variant scans (VARIATION_CEILING_RESEARCH
            // §3.ג). Guarded by `!truncated` on purpose — typo expansion has
            // the lowest priority, so when the exact branches alone exhaust a
            // budget (an extremely common word) the scan is skipped entirely
            // rather than pushed past the budget; the query then behaves as
            // if typo tolerance found nothing, and the warn! below records it.
            let builder = LevenshteinAutomatonBuilder::new(1, true);
            'typo: for token in typo_tokens {
                let automaton = DfaWrapper(builder.build_dfa(token));
                for inverted in &inverted_indexes {
                    if Self::collect_automaton_terms(
                        inverted,
                        &automaton,
                        max_expansions,
                        postings_budget,
                        &mut matched,
                        &mut postings_cost,
                    )? {
                        truncated = true;
                        break 'typo;
                    }
                }
            }
        }
        if truncated {
            warn!(
                "term collection truncated at {} terms / ~{postings_cost} postings \
                 (caps: {max_expansions} terms, {postings_budget} postings); \
                 serving the highest-priority branches collected so far",
                matched.len(),
            );
        }
        let terms: Arc<Vec<Term>> = Arc::new(
            matched
                .into_iter()
                .map(|t| Term::from_field_text(text_field, &t))
                .collect(),
        );
        let entry = CachedTermSet { terms, truncated };
        self.term_cache
            .lock()
            .unwrap()
            .put(cache_key, entry.clone());
        Ok(entry)
    }

    /// Streams every term `automaton` matches in one segment's dictionary
    /// into `matched`, charging each hit's per-segment `doc_freq` against
    /// `postings_cost` (the streamer decodes `TermInfo` in-memory on
    /// `advance()`, so reading the value adds no IO). Returns `true` when a
    /// collection budget was hit — degrade, never error: the check runs after
    /// insertion, so even a first term costlier than the whole budget is
    /// kept and the caller serves what was gathered.
    fn collect_automaton_terms<A>(
        inverted: &tantivy::InvertedIndexReader,
        automaton: &A,
        max_expansions: u32,
        postings_budget: u64,
        matched: &mut HashSet<String>,
        postings_cost: &mut u64,
    ) -> Result<bool>
    where
        A: Automaton,
        A::State: Clone,
    {
        let mut stream = inverted.terms().search(automaton).into_stream()?;
        while stream.advance() {
            if let Ok(term) = std::str::from_utf8(stream.key()) {
                *postings_cost += u64::from(stream.value().doc_freq);
                // contains-before-insert avoids re-allocating the term string
                // when it was already seen in an earlier segment or matched
                // by an earlier automaton.
                if !matched.contains(term) {
                    matched.insert(term.to_string());
                }
                if matched.len() >= max_expansions as usize || *postings_cost >= postings_budget {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Tokenizes `text` with the same `"hebrew"` analyzer the `text` field is
    /// indexed with — so exact/fuzzy terms line up with the (normalized) index
    /// term dictionary, including geresh/gershayim kept inside tokens (ז"ל,
    /// תוס'). No pre-normalization: the tokenizer already strips attached
    /// marks and folds presentation forms, and a `strip_nikud` pass here would
    /// also delete maqaf/sof-pasuq, gluing `"אשר־שמע"` into one bogus term.
    fn index_token_texts(&self, text: &str) -> Result<Vec<String>> {
        self.index_token_texts_with("hebrew_query", text)
    }

    /// [`Self::index_token_texts`] with an explicit analyzer —
    /// `"hebrew_vocalized"` tokenizes a vocalized query exactly like the
    /// `textVocalized` field (marks kept inside tokens).
    fn index_token_texts_with(&self, analyzer_name: &str, text: &str) -> Result<Vec<String>> {
        let mut analyzer = self
            .index
            .tokenizers()
            .get(analyzer_name)
            .with_context(|| format!("{analyzer_name} tokenizer not registered"))?;
        let mut stream = analyzer.token_stream(text);
        let mut out = Vec::new();
        while let Some(token) = stream.next() {
            out.push(token.text.clone());
        }
        Ok(out)
    }

    /// The stored/indexed text field a search reads: `textVocalized` when a
    /// vocalized flag is on, the plain `text` field otherwise.
    fn search_text_field(&self, voc: &VocalizedFlags) -> Result<Field> {
        if voc.any() {
            Ok(self.schema.get_field("textVocalized")?)
        } else {
            Ok(self.schema.get_field("text")?)
        }
    }

    /// Facet filter sub-query over the `topics` facet field.
    ///
    /// הנתיבים מחולקים לקבוצות לפי המקטע הראשון: כל שורש ממדי
    /// ([`FACET_DIMENSION_ROOTS`] — `author`/`era`/`base`) הוא קבוצה
    /// משלו, וכל השאר (עץ הקטגוריות/ספרים) קבוצה אחת. בתוך קבוצה —
    /// `TermSetQuery` (OR, בהתאמת-קידומת של facet); בין קבוצות — AND.
    /// כך "תקופת ראשונים AND המדף הנבחר" מתנהג נכון, בעוד קריאה עם
    /// נתיבי קטגוריות בלבד שקולה בדיוק להתנהגות הקודמת (קבוצה אחת).
    fn facet_filter_query(&self, facets: &[String]) -> Result<Box<dyn Query>> {
        let topics_f = self.schema.get_field("topics")?;
        // הסדר דטרמיניסטי: קטגוריות תחילה ואז הממדים לפי סדר ההגדרה —
        // אינדקס 0 = קטגוריות, i+1 = הממד ה-i.
        let mut groups: Vec<Vec<Term>> = vec![Vec::new(); FACET_DIMENSION_ROOTS.len() + 1];
        for f in facets {
            let facet = Facet::from_text(f)?;
            let root = f.trim_start_matches('/').split('/').next().unwrap_or("");
            let group = FACET_DIMENSION_ROOTS
                .iter()
                .position(|d| *d == root)
                .map_or(0, |i| i + 1);
            groups[group].push(Term::from_facet(topics_f, &facet));
        }
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = groups
            .into_iter()
            .filter(|terms| !terms.is_empty())
            .map(|terms| {
                (
                    Occur::Must,
                    Box::new(TermSetQuery::new(terms)) as Box<dyn Query>,
                )
            })
            .collect();
        Ok(match clauses.len() {
            // אין facets — פילטר ריק לא אמור להיקרא, אבל אם כן: הכל עובר.
            0 => Box::new(AllQuery),
            1 => clauses.pop().expect("one clause").1,
            _ => Box::new(BooleanQuery::new(clauses)),
        })
    }

    /// Exact mode: a `TermQuery` (one token) or `PhraseQuery` (several), filtered
    /// by facets. No regex — fastest path. With a vocalized flag on, the query
    /// runs against the `textVocalized` dictionary instead: each token becomes
    /// a required-marks regex ([`hebrew_query::vocalized_token_pattern`]), a
    /// single word materializes via [`Self::single_regex_term_query`] (which
    /// may truncate — the returned flag), a phrase goes through
    /// [`Self::phrase_query_with_degrade`].
    fn build_exact_query(
        &self,
        query_str: &str,
        facets: &[String],
        voc: &VocalizedFlags,
    ) -> Result<(Box<dyn Query>, bool)> {
        if voc.any() {
            return self.build_exact_query_vocalized(query_str, facets, voc);
        }
        let text_f = self.schema.get_field("text")?;
        let token_texts = self.index_token_texts(query_str)?;
        let mut terms: Vec<Term> = token_texts
            .iter()
            .map(|t| Term::from_field_text(text_f, t))
            .collect();
        let main_query: Box<dyn Query> = match terms.len() {
            0 => Box::new(EmptyQuery),
            1 => Box::new(TermQuery::new(
                terms.pop().unwrap(),
                IndexRecordOption::Basic,
            )),
            _ => Box::new(PhraseQuery::new(terms)),
        };
        if facets.is_empty() {
            Ok((main_query, false))
        } else {
            Ok((
                Box::new(BooleanQuery::new(vec![
                    (Occur::Must, main_query),
                    (Occur::Must, self.facet_filter_query(facets)?),
                ])),
                false,
            ))
        }
    }

    /// The vocalized arm of [`Self::build_exact_query`].
    fn build_exact_query_vocalized(
        &self,
        query_str: &str,
        facets: &[String],
        voc: &VocalizedFlags,
    ) -> Result<(Box<dyn Query>, bool)> {
        let voc_field = self.schema.get_field("textVocalized")?;
        let tokens = self.index_token_texts_with("hebrew_vocalized_query", query_str)?;
        let patterns: Vec<String> = tokens
            .iter()
            .map(|t| hebrew_query::vocalized_token_pattern(t, voc))
            .collect();
        let (main_query, truncated): (Box<dyn Query>, bool) = match patterns.len() {
            0 => (Box::new(EmptyQuery), false),
            1 => self.single_regex_term_query(
                &patterns,
                &[],
                voc_field,
                VOC_EXACT_SINGLE_MAX_EXPANSIONS,
            )?,
            _ => {
                // Free-mark runs make every vocalized token an expansion, so
                // a phrase of common words can outgrow the cumulative
                // `RegexPhraseQuery` ceiling; the shared degrade path keeps
                // the historical query when the counts fit and falls back to
                // the term-list phrase instead of erroring when they don't.
                let word_patterns: Vec<hebrew_query::WordPattern> = patterns
                    .iter()
                    .map(|p| hebrew_query::WordPattern::parse(p))
                    .collect();
                let gaps = vec![0u32; word_patterns.len().saturating_sub(1)];
                self.phrase_query_with_degrade(
                    &word_patterns,
                    &gaps,
                    voc_field,
                    VOC_PHRASE_MAX_EXPANSIONS,
                )?
            }
        };
        if facets.is_empty() {
            Ok((main_query, truncated))
        } else {
            Ok((
                Box::new(BooleanQuery::new(vec![
                    (Occur::Must, main_query),
                    (Occur::Must, self.facet_filter_query(facets)?),
                ])),
                truncated,
            ))
        }
    }

    /// Expands vocalized typo/fuzzy tokens: scans the PLAIN dictionary with a
    /// Levenshtein automaton over each mark-free base (edit distance over
    /// marked terms would count every mark as an edit), and returns one
    /// free-mark branch per *existing* variant, re-projected onto the
    /// vocalized dictionary. The base itself is skipped — the caller already
    /// carries it as the required-marks exact branch, and a free duplicate
    /// would silently erase the typed-marks constraint.
    fn vocalized_variant_branches(
        &self,
        searcher: &Searcher,
        bases: &[String],
        distance: u8,
        seen: &mut HashSet<String>,
    ) -> Result<Vec<String>> {
        if bases.is_empty() || distance == 0 {
            return Ok(Vec::new());
        }
        let plain_field = self.schema.get_field("text")?;
        let builder = LevenshteinAutomatonBuilder::new(distance, true);
        let mut branches = Vec::new();
        for base in bases {
            let automaton = DfaWrapper(builder.build_dfa(base));
            for variant in self.automaton_terms_in_field(
                searcher,
                plain_field,
                &automaton,
                VOC_VARIANTS_PER_TOKEN,
            )? {
                if &variant == base || !seen.insert(variant.clone()) {
                    continue;
                }
                branches.push(hebrew_query::vocalized_free_pattern(&variant));
                if branches.len() >= VOC_VARIANTS_PER_TOKEN {
                    return Ok(branches);
                }
            }
        }
        Ok(branches)
    }

    /// The vocalized arm of the approximate (`fuzzy`) mode. Per token, the
    /// branch list runs highest-priority-first (the collection contract of
    /// [`Self::single_regex_term_query`]): the exact form with its typed
    /// marks REQUIRED, then the quote-free spelling, then the lexicon's
    /// morphological relatives, then existing edit-distance variants — the
    /// last three mark-free (their letters differ from what was typed, so
    /// the typed marks have no positions to attach to). Each token
    /// materializes into a `TermSetQuery` over `textVocalized`; multi-word
    /// queries AND the tokens like the plain fuzzy path (documents holding
    /// all words anywhere — no phrase constraint). No relevance tiers: the
    /// vocalized paths order by catalogue, and an unranked recall query pays
    /// nothing for scoring it never uses.
    fn build_fuzzy_query_vocalized(
        &self,
        query_str: &str,
        facets: &[String],
        max_distance: u8,
        voc: &VocalizedFlags,
    ) -> Result<(Box<dyn Query>, bool)> {
        anyhow::ensure!(
            max_distance <= 2,
            "fuzzy distance is limited to 2, got {max_distance}"
        );
        let tokens = self.index_token_texts_with("hebrew_vocalized_query", query_str)?;
        if tokens.is_empty() {
            return Ok((Box::new(EmptyQuery), false));
        }
        let voc_field = self.schema.get_field("textVocalized")?;
        let searcher = self.index_reader.searcher();
        let mut truncated = false;
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(tokens.len() + 1);
        for token in &tokens {
            let base = hebrew_query::strip_attached_marks(token);
            let mut branches: Vec<String> = vec![hebrew_query::vocalized_token_pattern(token, voc)];
            let mut seen: HashSet<String> = HashSet::from([base.clone()]);
            if let Some(clean) = Self::quoteless_variant(&base) {
                if seen.insert(clean.clone()) {
                    branches.push(hebrew_query::vocalized_free_pattern(&clean));
                }
            }
            if max_distance > 0 {
                if let Some(dict) = self.magic_dict.as_ref() {
                    for form in dict.recall_forms(&base, MAX_LEXICAL_FORMS) {
                        if seen.insert(form.clone()) {
                            branches.push(hebrew_query::vocalized_free_pattern(&form));
                        }
                    }
                }
                branches.extend(self.vocalized_variant_branches(
                    &searcher,
                    std::slice::from_ref(&base),
                    max_distance,
                    &mut seen,
                )?);
            }
            let (token_query, token_truncated) =
                self.single_regex_term_query(&branches, &[], voc_field, VOC_FUZZY_MAX_EXPANSIONS)?;
            truncated |= token_truncated;
            clauses.push((Occur::Must, token_query));
        }
        if !facets.is_empty() {
            clauses.push((Occur::Must, self.facet_filter_query(facets)?));
        }
        let query: Box<dyn Query> = if clauses.len() == 1 {
            clauses.pop().expect("one clause").1
        } else {
            Box::new(BooleanQuery::new(clauses))
        };
        Ok((query, truncated))
    }

    /// The quote-free spelling of a token that carries gershayim/geresh
    /// (`רמב"ם` → `רמבם`). Clean-typography editions store that term, so the
    /// fuzzy builders add it as an exact-tier alternative: the bridge works
    /// even at distance 0, and clean-edition hits rank as exact matches
    /// instead of edit-distance tail matches. `None` when the token has no
    /// quotes (the common case) or nothing remains without them.
    fn quoteless_variant(token: &str) -> Option<String> {
        if !token.contains(['"', '\'']) {
            return None;
        }
        let clean: String = token.chars().filter(|c| !matches!(c, '"' | '\'')).collect();
        (!clean.is_empty()).then_some(clean)
    }

    /// The two summed `Should` clauses that lift an exact-token hit to the top
    /// relevance tier: a `ConstScoreQuery` floor (immune to BM25 `idf` collapse
    /// on near-ubiquitous terms) plus a small BM25 `TermQuery` add-on for
    /// intra-exact ordering. Only used on the ranked (`Relevance`) fuzzy path.
    fn exact_rank_clauses(text_f: Field, token: &str) -> Vec<(Occur, Box<dyn Query>)> {
        let term = Term::from_field_text(text_f, token);
        vec![
            (
                Occur::Should,
                Box::new(ConstScoreQuery::new(
                    Box::new(TermQuery::new(term.clone(), IndexRecordOption::Basic)),
                    FUZZY_BOOST_EXACT,
                )) as Box<dyn Query>,
            ),
            (
                Occur::Should,
                Box::new(BoostQuery::new(
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)),
                    FUZZY_BOOST_EXACT_REL,
                )),
            ),
        ]
    }

    /// Fuzzy mode from pre-tokenized terms: one `FuzzyTermQuery` per term, ANDed,
    /// filtered by facets. `rank` adds the exact relevance tier (see
    /// [`Self::exact_rank_clauses`]); count/catalogue paths pass `false`.
    fn build_fuzzy_query_from_terms(
        &self,
        term_texts: &[String],
        facets: &[String],
        max_distance: u8,
        rank: bool,
    ) -> Result<Box<dyn Query>> {
        // Tantivy only rejects distances above 2 when the query executes
        // (InvalidArgument from FuzzyTermQuery's weight); validate upfront so
        // every fuzzy path fails fast with a clear error instead.
        anyhow::ensure!(
            max_distance <= 2,
            "fuzzy distance is limited to 2, got {max_distance}"
        );
        // Mirror exact mode: an empty query matches nothing. Without this
        // guard the clause list degenerates to just the facet filter and the
        // query returns every document in the selected facets.
        if term_texts.is_empty() {
            return Ok(Box::new(EmptyQuery));
        }
        let text_f = self.schema.get_field("text")?;
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = term_texts
            .iter()
            .map(|t| {
                let term = Term::from_field_text(text_f, t);
                let fuzzy = FuzzyTermQuery::new(term, max_distance, true);
                // Bare recall is one fuzzy automaton per token. distance 0 is
                // exact already, and unranked paths (count/catalogue) need no
                // scoring, so both stay the bare query — except that a
                // quote-bearing token also matches its quote-free spelling
                // (clean-typography editions), even at distance 0. On the
                // ranked path above distance 0 we add the exact tier so an
                // exact hit outranks a bare edit-distance neighbour; the exact
                // term is a subset of the fuzzy match, so recall is unchanged.
                let token_query: Box<dyn Query> = if !rank || max_distance == 0 {
                    match Self::quoteless_variant(t) {
                        Some(clean) => Box::new(BooleanQuery::new(vec![
                            (Occur::Should, Box::new(fuzzy) as Box<dyn Query>),
                            (
                                Occur::Should,
                                Box::new(TermQuery::new(
                                    Term::from_field_text(text_f, &clean),
                                    IndexRecordOption::Basic,
                                )),
                            ),
                        ])),
                        None => Box::new(fuzzy),
                    }
                } else {
                    let mut should = Self::exact_rank_clauses(text_f, t);
                    if let Some(clean) = Self::quoteless_variant(t) {
                        should.extend(Self::exact_rank_clauses(text_f, &clean));
                    }
                    should.push((
                        Occur::Should,
                        Box::new(BoostQuery::new(Box::new(fuzzy), FUZZY_BOOST_FUZZY)),
                    ));
                    Box::new(BooleanQuery::new(should))
                };
                (Occur::Must, token_query)
            })
            .collect();
        if !facets.is_empty() {
            clauses.push((Occur::Must, self.facet_filter_query(facets)?));
        }
        Ok(Box::new(BooleanQuery::new(clauses)))
    }

    /// Fuzzy mode from a raw query string (tokenized like the index). Used only
    /// by the count/facet paths, which never rank — hence `rank: false`.
    fn build_fuzzy_query(
        &self,
        query: &str,
        facets: &[String],
        max_distance: u8,
    ) -> Result<Box<dyn Query>> {
        let token_texts = self.index_token_texts(query)?;
        self.build_fuzzy_search_query(&token_texts, facets, max_distance, false)
    }

    /// Approximate (`fuzzy`) recall query. Routes through the lexical builder
    /// when a `MagicDictionary` is loaded, otherwise the plain fuzzy builder.
    /// This is the single decision point so every fuzzy entry point
    /// (`search_*`/`count_*`) shares identical matching logic. `rank` toggles
    /// the relevance-scoring layer: `true` only for `ResultsOrder::Relevance`
    /// searches, `false` for counts and catalogue ordering (which ignore score)
    /// so they build the bare recall query and pay nothing for unused ranking.
    fn build_fuzzy_search_query(
        &self,
        term_texts: &[String],
        facets: &[String],
        max_distance: u8,
        rank: bool,
    ) -> Result<Box<dyn Query>> {
        if self.magic_dict.is_some() && max_distance > 0 {
            self.build_lexical_fuzzy_query(term_texts, facets, max_distance, rank)
        } else {
            self.build_fuzzy_query_from_terms(term_texts, facets, max_distance, rank)
        }
    }

    /// Lexical fuzzy mode: per token, `(FuzzyTermQuery OR TermSetQuery[lexical
    /// forms])` is required (`MUST`); the inner `SHOULD` group keeps both
    /// edit-distance matches and morphological relatives. Falls back to the
    /// bare fuzzy clause for tokens the dictionary doesn't know. Facets filter
    /// as usual. Independent of exact/advanced — only the fuzzy path calls it.
    fn build_lexical_fuzzy_query(
        &self,
        term_texts: &[String],
        facets: &[String],
        max_distance: u8,
        rank: bool,
    ) -> Result<Box<dyn Query>> {
        anyhow::ensure!(
            max_distance <= 2,
            "fuzzy distance is limited to 2, got {max_distance}"
        );
        if term_texts.is_empty() {
            return Ok(Box::new(EmptyQuery));
        }
        let dict = self
            .magic_dict
            .as_ref()
            .context("lexical fuzzy query requires a loaded magic dictionary")?;
        let text_f = self.schema.get_field("text")?;

        if term_texts.len() > 1 {
            let patterns = self.lexical_fuzzy_phrase_patterns(dict, term_texts, max_distance)?;
            let mut phrase_query = RegexPhraseQuery::new(text_f, patterns);
            phrase_query.set_slop(LEXICAL_FUZZY_PHRASE_SLOP);
            phrase_query
                .set_max_expansions((MAX_LEXICAL_PHRASE_TERMS_PER_TOKEN * term_texts.len()) as u32);
            let main_query: Box<dyn Query> = Box::new(phrase_query);
            return if facets.is_empty() {
                Ok(main_query)
            } else {
                Ok(Box::new(BooleanQuery::new(vec![
                    (Occur::Must, main_query),
                    (Occur::Must, self.facet_filter_query(facets)?),
                ])))
            };
        }

        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(term_texts.len() + 1);
        for token in term_texts {
            let exact_term = Term::from_field_text(text_f, token);
            // Wrap the fuzzy automaton in the fuzzy-tier boost only when ranking;
            // an unranked recall query (count/catalogue) carries no boost so it
            // stays the bare `FuzzyTermQuery` it always was.
            let fuzzy_q = FuzzyTermQuery::new(exact_term, max_distance, true);
            let fuzzy: Box<dyn Query> = if rank {
                Box::new(BoostQuery::new(Box::new(fuzzy_q), FUZZY_BOOST_FUZZY))
            } else {
                Box::new(fuzzy_q)
            };
            let clean = Self::quoteless_variant(token);
            let mut forms = dict.recall_forms(token, MAX_LEXICAL_FORMS);
            forms.retain(|f| f != token && Some(f.as_str()) != clean.as_deref());

            // Unranked: the original recall shape — `fuzzy OR termset`, or just
            // `fuzzy` when the dictionary has no extra forms. Ranked: prepend the
            // exact tier (the exact term is a subset of the fuzzy match, so this
            // never changes recall) and boost the lexical tier. `BooleanQuery`
            // sums `Should` scores, so exact-floor + BM25 > lexical > fuzzy.
            // A quote-bearing token also carries its quote-free spelling in
            // the exact tier — clean-typography editions match at distance 0
            // and rank as exact, not as edit-distance tail.
            let mut should: Vec<(Occur, Box<dyn Query>)> = if rank {
                Self::exact_rank_clauses(text_f, token)
            } else {
                Vec::with_capacity(3)
            };
            if let Some(clean) = &clean {
                if rank {
                    should.extend(Self::exact_rank_clauses(text_f, clean));
                } else {
                    should.push((
                        Occur::Should,
                        Box::new(TermQuery::new(
                            Term::from_field_text(text_f, clean),
                            IndexRecordOption::Basic,
                        )),
                    ));
                }
            }
            should.push((Occur::Should, fuzzy));
            if !forms.is_empty() {
                let set_terms: Vec<Term> = forms
                    .iter()
                    .map(|f| Term::from_field_text(text_f, f))
                    .collect();
                let termset: Box<dyn Query> = if rank {
                    Box::new(BoostQuery::new(
                        Box::new(TermSetQuery::new(set_terms)),
                        FUZZY_BOOST_LEXICAL,
                    ))
                } else {
                    Box::new(TermSetQuery::new(set_terms))
                };
                should.push((Occur::Should, termset));
            }

            // A single bare `FuzzyTermQuery` (no forms, unranked) needs no
            // wrapping `BooleanQuery` — keep it byte-identical to the original.
            let token_query: Box<dyn Query> = if should.len() == 1 {
                should.pop().unwrap().1
            } else {
                Box::new(BooleanQuery::new(should))
            };
            clauses.push((Occur::Must, token_query));
        }
        if !facets.is_empty() {
            clauses.push((Occur::Must, self.facet_filter_query(facets)?));
        }
        Ok(Box::new(BooleanQuery::new(clauses)))
    }

    fn lexical_fuzzy_phrase_patterns(
        &self,
        dict: &MagicDictionary,
        term_texts: &[String],
        max_distance: u8,
    ) -> Result<Vec<String>> {
        let builder = LevenshteinAutomatonBuilder::new(max_distance, true);
        // Query-time enumeration (not highlight) — no search-scoped searcher
        // exists yet, so take a fresh one like the other query builders do.
        let searcher = self.index_reader.searcher();
        term_texts
            .iter()
            .map(|token| {
                let mut terms = Vec::new();
                let mut seen = HashSet::new();

                Self::push_limited_unique(
                    &mut terms,
                    &mut seen,
                    token.clone(),
                    MAX_LEXICAL_PHRASE_TERMS_PER_TOKEN,
                );
                // The quote-free spelling rides along ahead of the budgeted
                // expansions, like in the single-token path.
                if let Some(clean) = Self::quoteless_variant(token) {
                    Self::push_limited_unique(
                        &mut terms,
                        &mut seen,
                        clean,
                        MAX_LEXICAL_PHRASE_TERMS_PER_TOKEN,
                    );
                }
                for form in dict.recall_forms(token, MAX_LEXICAL_FORMS) {
                    Self::push_limited_unique(
                        &mut terms,
                        &mut seen,
                        form,
                        MAX_LEXICAL_PHRASE_TERMS_PER_TOKEN,
                    );
                }

                let remaining = MAX_LEXICAL_PHRASE_TERMS_PER_TOKEN.saturating_sub(terms.len());
                if remaining > 0 {
                    let automaton = DfaWrapper(builder.build_dfa(token));
                    for fuzzy_term in self.automaton_terms(&searcher, &automaton, remaining)? {
                        Self::push_limited_unique(
                            &mut terms,
                            &mut seen,
                            fuzzy_term,
                            MAX_LEXICAL_PHRASE_TERMS_PER_TOKEN,
                        );
                    }
                }

                Ok(Self::terms_regex_union(&terms))
            })
            .collect()
    }

    fn push_limited_unique(
        out: &mut Vec<String>,
        seen: &mut HashSet<String>,
        value: String,
        cap: usize,
    ) {
        if out.len() < cap && seen.insert(value.clone()) {
            out.push(value);
        }
    }

    fn terms_regex_union(terms: &[String]) -> String {
        if terms.len() == 1 {
            return Self::escape_regex_term(&terms[0]);
        }
        let escaped = terms
            .iter()
            .map(|term| Self::escape_regex_term(term))
            .collect::<Vec<_>>();
        format!("(?:{})", escaped.join("|"))
    }

    fn escape_regex_term(term: &str) -> String {
        let mut out = String::with_capacity(term.len());
        for ch in term.chars() {
            if matches!(
                ch,
                '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
            ) {
                out.push('\\');
            }
            out.push(ch);
        }
        out
    }

    /// Advanced mode: ports the Dart morphological query builder to produce regex
    /// terms + slop + max_expansions, then reuses `build_query`. Also returns the
    /// regex patterns so callers can materialize concrete terms for highlighting.
    fn build_advanced_query(
        &self,
        query: &str,
        distance: u32,
        custom_spacing: &HashMap<String, String>,
        alternative_words: &HashMap<u32, Vec<String>>,
        search_options: &HashMap<String, HashMap<String, bool>>,
        facets: Vec<String>,
        voc: &VocalizedFlags,
        scope: &SearchScope,
        match_mode: &WordMatch,
    ) -> Result<AdvancedQueryBuild> {
        // The vocalized mode is requested either by the global API flags or
        // by a per-word "ניקוד"/"טעמים" option; the per-word requirement
        // derivation happens inside `prepare_advanced_query_vocalized`, which
        // receives the GLOBAL flags only (folding options into them would
        // bind every word's typed marks).
        let voc_mode = voc.or(hebrew_query::options_vocalized_flags(search_options));
        // "תרגום ארמי": מילה שסומנה לה האפשרות מקבלת את תרגומיה מהמילון
        // כמילים-חלופיות — ומשם הן זורמות בכל המסלולים הקיימים (ענפי
        // תבנית, המסלול המנוקד, איסוף טרמים להדגשה).
        let translated =
            self.translation_alternatives(query, alternative_words, search_options, &voc_mode);
        let alternative_words = translated.as_ref().unwrap_or(alternative_words);
        let mut prepared = if voc_mode.any() {
            hebrew_query::prepare_advanced_query_vocalized(
                query,
                distance,
                custom_spacing,
                alternative_words,
                search_options,
                voc,
            )
        } else {
            hebrew_query::prepare_advanced_query(
                query,
                distance,
                custom_spacing,
                alternative_words,
                search_options,
            )
        };
        let text_field = self.search_text_field(&voc_mode)?;
        // Vocalized typo tokens cannot ride the in-collection Levenshtein
        // scan (it would run on the vocalized dictionary, counting every mark
        // as an edit): expand them against the PLAIN dictionary here and
        // append the variants as lowest-priority free-mark branches.
        if voc_mode.any() && !prepared.typo_tokens.is_empty() {
            let searcher = self.index_reader.searcher();
            let mut seen: HashSet<String> = prepared.typo_tokens.iter().cloned().collect();
            let extra =
                self.vocalized_variant_branches(&searcher, &prepared.typo_tokens, 1, &mut seen)?;
            prepared.typo_tokens = Vec::new();
            if let Some(first) = prepared.regex_terms.pop() {
                prepared.regex_terms.push(first.with_extra_branches(extra));
            }
        }
        // `gaps` already folds `custom_spacing` in (per-pair values, else
        // `distance` for every pair) — the phrase filter's gap allowances
        // must use it, not the raw `distance`, or a spacing-permitted match
        // would be rejected and fall back to the broad term highlight.
        let gaps = prepared.gaps.clone();
        let source_words = std::mem::take(&mut prepared.words);
        // The highlight builders want one pattern string per word; the query
        // builder gets the structured patterns so a single word compiles per
        // branch instead of as one state-limited DFA.
        let regex_terms: Vec<String> = prepared
            .regex_terms
            .iter()
            .map(hebrew_query::WordPattern::joined)
            .collect();
        // "ראשי תיבות": תת-שאילתות פענוח ר"ת שיש ל-OR עם השאילתה הראשית.
        // נבנות עכשיו — לפני ש-`query` (מחרוזת) מוצללת ע"י תוצאת בניית
        // השאילתה — ובאותם facets/scope כדי שה-OR יישאר מפולטר נכון.
        let acronym_alts = self.acronym_alternatives(
            query,
            search_options,
            &voc_mode,
            facets.clone(),
            prepared.max_expansions,
            text_field,
            scope,
        )?;
        let (main_query, main_truncated) = self.build_query_from_patterns(
            prepared.regex_terms,
            &source_words,
            &prepared.typo_tokens,
            facets,
            &gaps,
            prepared.max_expansions,
            text_field,
            scope,
            match_mode,
        )?;
        let (query, truncated, acronym_patterns) = if acronym_alts.is_empty() {
            (main_query, main_truncated, Vec::new())
        } else {
            let mut truncated = main_truncated;
            let mut clauses: Vec<(Occur, Box<dyn Query>)> =
                Vec::with_capacity(acronym_alts.len() + 1);
            let mut alt_patterns = Vec::with_capacity(acronym_alts.len());
            clauses.push((Occur::Should, main_query));
            for (alt_query, alt_truncated, patterns) in acronym_alts {
                clauses.push((Occur::Should, alt_query));
                truncated |= alt_truncated;
                alt_patterns.push(patterns);
            }
            (
                Box::new(BooleanQuery::new(clauses)) as Box<dyn Query>,
                truncated,
                alt_patterns,
            )
        };
        Ok((query, regex_terms, gaps, truncated, acronym_patterns))
    }

    /// בונה תת-שאילתות פענוח ראשי-תיבות (דו-כיווני) שיש ל-OR עם השאילתה
    /// הראשית, כשאפשרות "ראשי תיבות" ([`hebrew_query::OPT_ACRONYM`]) דלוקה
    /// על מילה כלשהי. מחזיר וקטור ריק כשאין מילון, אין אפשרות מסומנת, או
    /// אין התאמה.
    ///
    /// הפענוח **רב-מילי**, ולכן אינו יכול לרכוב על ערוץ `alternative_words`
    /// (החד-מילתי) כמו התרגום — כל חלופה נבנית כשאילתה שלמה ומצטרפת כ-OR.
    /// הכיסוי מוגבל ל**שאילתה שהיא יחידה סמנטית אחת**: ר"ת בודד (כיוון
    /// ר"ת→פענוח) או ביטוי שכולו פענוח ידוע (כיוון פענוח→ר"ת). ר"ת המשובץ
    /// בתוך שאילתה ארוכה יותר, ומצב מנוקד, אינם נתמכים בשלב זה.
    ///
    /// כל פריט מוחזר כ-(שאילתה, truncated, תבניות-ליטרל פר-מילה) — התבניות
    /// מוזנות לבוני ההדגשה כדי שמסמך שנמצא דרך החלופה ייצבע.
    #[allow(clippy::too_many_arguments)]
    fn acronym_alternatives(
        &self,
        query: &str,
        search_options: &HashMap<String, HashMap<String, bool>>,
        voc: &VocalizedFlags,
        facets: Vec<String>,
        max_expansions: u32,
        text_field: Field,
        scope: &SearchScope,
    ) -> Result<Vec<(Box<dyn Query>, bool, Vec<String>)>> {
        let Some(dict) = self.acronym_dict.as_ref() else {
            return Ok(Vec::new());
        };
        if search_options.is_empty() {
            return Ok(Vec::new());
        }
        // פענוח בשילוב חיפוש מנוקד אינו נתמך עדיין: החלופות ליטרלים
        // נטולי-סימנים ולא יתאימו למילון הטרמים המנוקד.
        if voc.any() {
            return Ok(Vec::new());
        }
        // אותה נורמליזציה/טוקניזציה שממנה נגזרים מפתחות האפשרויות ומפתחות
        // המילון — כדי שהכול יתלכד.
        let words = hebrew_query::split_query_words(&hebrew_query::normalize_for_index(query));
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let enabled = words.iter().enumerate().any(|(i, word)| {
            search_options
                .get(&format!("{word}_{i}"))
                .and_then(|opts| opts.get(hebrew_query::OPT_ACRONYM))
                .copied()
                .unwrap_or(false)
        });
        if !enabled {
            return Ok(Vec::new());
        }

        // אוסף החלופות כרשימות-מילים בצורת טרם-אינדקס.
        let mut alternatives: Vec<Vec<String>> = Vec::new();
        // כיוון א' (ר"ת→פענוח): רק כשהשאילתה כולה ר"ת בודד.
        if words.len() == 1 {
            alternatives.extend(dict.expand(&words[0], MAX_ACRONYM_EXPANSIONS));
        }
        // כיוון ב' (פענוח→ר"ת): כשכל השאילתה היא פענוח ידוע.
        for acronym in dict.acronyms_for(&words, MAX_ACRONYM_EXPANSIONS) {
            alternatives.push(vec![acronym]);
        }
        if alternatives.is_empty() {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(alternatives.len());
        for alt in alternatives {
            let literal_patterns: Vec<String> =
                alt.iter().map(|w| hebrew_query::escape_regex(w)).collect();
            let patterns: Vec<hebrew_query::WordPattern> = literal_patterns
                .iter()
                .map(|p| hebrew_query::WordPattern::Literal(p.clone()))
                .collect();
            // פענוח הוא ביטוי קנוני — מילים צמודות (slop 0, ללא GapVerified),
            // וכל מילותיו חובה גם כשהשאילתה הראשית בהתאמה חלקית.
            let gaps = vec![0u32; patterns.len().saturating_sub(1)];
            let (alt_query, truncated) = self.build_query_from_patterns(
                patterns,
                &[],
                &[],
                facets.clone(),
                &gaps,
                max_expansions,
                text_field,
                scope,
                &WordMatch::All,
            )?;
            out.push((alt_query, truncated, literal_patterns));
        }
        Ok(out)
    }

    /// בונה מפת מילים-חלופיות מורחבת בתרגומי המילון עבור מילים שסומנה
    /// להן אפשרות "תרגום ארמי". מחזיר `None` כשאין מה להרחיב (אין מילון,
    /// אין אפשרות מסומנת, או אין תרגומים) — והשאילתה ממשיכה עם המפה
    /// המקורית ללא העתקה.
    fn translation_alternatives(
        &self,
        query: &str,
        alternative_words: &HashMap<u32, Vec<String>>,
        search_options: &HashMap<String, HashMap<String, bool>>,
        voc: &VocalizedFlags,
    ) -> Option<HashMap<u32, Vec<String>>> {
        let dict = self.translation_dict.as_ref()?;
        if search_options.is_empty() {
            return None;
        }
        // אותה נורמליזציה וטוקניזציה שממנה נגזרים מפתחות האפשרויות
        // ("{word}_{index}") בהכנת השאילתה — כדי שהמפתחות יתלכדו.
        let normalized = if voc.any() {
            hebrew_query::normalize_for_index_vocalized(query)
        } else {
            hebrew_query::normalize_for_index(query)
        };
        let words = hebrew_query::split_query_words(&normalized);
        let mut augmented: Option<HashMap<u32, Vec<String>>> = None;
        for (i, word) in words.iter().enumerate() {
            let enabled = search_options
                .get(&format!("{word}_{i}"))
                .and_then(|opts| opts.get(hebrew_query::OPT_TRANSLATION))
                .copied()
                .unwrap_or(false);
            if !enabled {
                continue;
            }
            // המילון ממופתח בצורת טרם נטולת-סימנים; במצב מנוקד המילה עוד
            // נושאת את סימניה.
            let base = hebrew_query::strip_attached_marks(word);
            let expansions = dict.expansions(&base, MAX_TRANSLATION_EXPANSIONS);
            if expansions.is_empty() {
                continue;
            }
            augmented
                .get_or_insert_with(|| alternative_words.clone())
                .entry(i as u32)
                .or_default()
                .extend(expansions);
        }
        augmented
    }

    // ── Shared query executors (take a prebuilt query) ───────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn run_search<F>(
        &self,
        query: Box<dyn Query>,
        make_highlight: F,
        text_field: Field,
        limit: u32,
        offset: u32,
        order: &ResultsOrder,
        hl: &HighlightConfig,
        grouping: Option<&ResultGrouping>,
    ) -> Result<Vec<SearchResult>>
    where
        F: FnOnce(&Searcher) -> Result<HighlightPlan>,
    {
        let searcher = self.index_reader.searcher();
        if let Some(grouping) = grouping {
            let page = Self::collect_grouped(&searcher, &*query, grouping, limit, offset, order)?;
            return Self::build_grouped_results(
                &self.schema,
                &searcher,
                &query,
                make_highlight,
                text_field,
                hl,
                &page,
            );
        }
        let addresses = Self::collect_addresses(&searcher, &*query, limit, offset, order)?;
        if addresses.is_empty() {
            return Ok(Vec::new());
        }
        let plan = Self::resolve_highlight(&searcher, make_highlight);
        let hl_q: &dyn Query = plan.query.as_deref().unwrap_or(query.as_ref());
        Self::build_results(
            &self.schema,
            &searcher,
            hl_q,
            text_field,
            addresses,
            hl,
            plan.phrase.as_ref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_search_and_count<F>(
        &self,
        query: Box<dyn Query>,
        make_highlight: F,
        text_field: Field,
        limit: u32,
        offset: u32,
        order: &ResultsOrder,
        hl: &HighlightConfig,
        truncated: bool,
        grouping: Option<&ResultGrouping>,
    ) -> Result<SearchPageResult>
    where
        F: FnOnce(&Searcher) -> Result<HighlightPlan>,
    {
        let searcher = self.index_reader.searcher();
        if let Some(grouping) = grouping {
            let page = Self::collect_grouped(&searcher, &*query, grouping, limit, offset, order)?;
            let results = Self::build_grouped_results(
                &self.schema,
                &searcher,
                &query,
                make_highlight,
                text_field,
                hl,
                &page,
            )?;
            return Ok(SearchPageResult {
                total_count: page.raw_total,
                results,
                truncated: truncated || page.truncated,
                group_count: Some(page.group_count),
            });
        }
        // Tuple collector: single index pass for both count and top-docs.
        let (addresses, total_count): (Vec<DocAddress>, u32) = match order {
            ResultsOrder::Catalogue => {
                let top_collector = TopDocs::with_limit(limit as usize)
                    .and_offset(offset as usize)
                    .order_by_fast_field::<u64>("id", Order::Asc);
                let (top_docs, count) = searcher.search(&*query, &(top_collector, Count))?;
                let addrs = top_docs.into_iter().map(|(_, addr)| addr).collect();
                (addrs, count as u32)
            }
            ResultsOrder::Generation => {
                let top_collector = TopDocs::with_limit(limit as usize)
                    .and_offset(offset as usize)
                    .order_by_fast_field::<u64>("generationSort", Order::Asc);
                let (top_docs, count) = searcher.search(&*query, &(top_collector, Count))?;
                let addrs = top_docs.into_iter().map(|(_, addr)| addr).collect();
                (addrs, count as u32)
            }
            ResultsOrder::Relevance => {
                let top_collector = TopDocs::with_limit(limit as usize)
                    .and_offset(offset as usize)
                    .order_by_score();
                let (top_docs, count) = searcher.search(&*query, &(top_collector, Count))?;
                let addrs = top_docs.into_iter().map(|(_, addr)| addr).collect();
                (addrs, count as u32)
            }
        };
        // total_count is the full hit count regardless of this page; only the
        // snippet highlighting (and its dictionary scan) is page-dependent, so
        // skip it when this page is empty (e.g. offset past the last hit).
        if addresses.is_empty() {
            return Ok(SearchPageResult {
                total_count,
                results: Vec::new(),
                truncated,
                group_count: None,
            });
        }
        let plan = Self::resolve_highlight(&searcher, make_highlight);
        let hl_q: &dyn Query = plan.query.as_deref().unwrap_or(query.as_ref());
        let results = Self::build_results(
            &self.schema,
            &searcher,
            hl_q,
            text_field,
            addresses,
            hl,
            plan.phrase.as_ref(),
        )?;
        Ok(SearchPageResult {
            total_count,
            results,
            truncated,
            group_count: None,
        })
    }

    fn run_count(&self, query: Box<dyn Query>) -> Result<u32> {
        let searcher = self.index_reader.searcher();
        Ok(searcher.search(&*query, &Count)? as u32)
    }

    fn run_count_by_book(&self, query: Box<dyn Query>) -> Result<HashMap<String, u32>> {
        let searcher = self.index_reader.searcher();
        Ok(searcher.search(&*query, &BookCountCollector)?)
    }

    fn run_facet_counts(
        &self,
        query: Box<dyn Query>,
        facet_prefix: String,
    ) -> Result<Vec<FacetCount>> {
        let searcher = self.index_reader.searcher();
        let mut facet_collector = FacetCollector::for_field("topics");
        facet_collector.add_facet(&facet_prefix);
        let facet_counts = searcher.search(&*query, &facet_collector)?;
        // FacetCounts::get<T> requires Facet: From<T>; &str satisfies this.
        let results = facet_counts
            .get(facet_prefix.as_str())
            .map(|(f, count)| FacetCount {
                path: f.to_string(),
                count,
            })
            .collect();
        Ok(results)
    }

    /// Routes a stream-search failure into the stream itself, where the Dart
    /// side receives it as an `onError` event.
    ///
    /// Returning `Err` from a `StreamSink`-taking function does NOT reach the
    /// app: the generated Dart wrapper fires the call `unawaited` and returns
    /// `sink.stream` immediately, so the error becomes an unhandled async
    /// error while dropping the Rust sink just closes the stream — the user
    /// sees 0 results and no failure. This was the silent-failure half of the
    /// state-limit bug, and relaxing budgets makes `max_expansions` overflow
    /// (which must stay an error) more reachable, so it has to be visible.
    fn surface_stream_error<T: crate::frb_generated::SseEncode>(
        sink: &StreamSink<T>,
        result: Result<()>,
    ) -> Result<()> {
        if let Err(err) = result {
            error!("stream search failed: {err:#}");
            let _ = sink.add_error(err);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_search_stream<F>(
        &self,
        query: Box<dyn Query>,
        make_highlight: F,
        text_field: Field,
        limit: u32,
        offset: u32,
        order: &ResultsOrder,
        hl: &HighlightConfig,
        chunk_size: u32,
        grouping: Option<&ResultGrouping>,
        sink: &StreamSink<Vec<SearchResult>>,
    ) -> Result<()>
    where
        F: FnOnce(&Searcher) -> Result<HighlightPlan>,
    {
        let searcher = self.index_reader.searcher();
        let chunk_size = (chunk_size.max(1)) as usize;
        if let Some(grouping) = grouping {
            let page = Self::collect_grouped(&searcher, &*query, grouping, limit, offset, order)?;
            let results = Self::build_grouped_results(
                &self.schema,
                &searcher,
                &query,
                make_highlight,
                text_field,
                hl,
                &page,
            )?;
            for chunk in results.chunks(chunk_size) {
                if sink.add(chunk.to_vec()).is_err() {
                    break;
                }
            }
            return Ok(());
        }
        let addresses = Self::collect_addresses(&searcher, &*query, limit, offset, order)?;
        if addresses.is_empty() {
            return Ok(());
        }
        let plan = Self::resolve_highlight(&searcher, make_highlight);
        let hl_q: &dyn Query = plan.query.as_deref().unwrap_or(query.as_ref());
        let phrase = plan.phrase.as_ref();
        // One generator for the whole stream: creating it resolves term
        // doc-frequencies, which is too expensive to repeat per chunk.
        let snippet_generator = Self::make_snippet_generator(&searcher, hl_q, text_field, hl)?;
        for chunk in addresses.chunks(chunk_size) {
            let results = Self::build_results_with_generator(
                &self.schema,
                &searcher,
                &snippet_generator,
                text_field,
                chunk.to_vec(),
                hl,
                phrase,
            )?;
            // If the Dart side cancelled the stream, stop early.
            if sink.add(results).is_err() {
                break;
            }
        }
        Ok(())
    }

    /// Combined-stream executor: ONE `searcher.search` pass evaluates the
    /// query with a tuple collector — ranked page + total count + per-book
    /// counts — then streams snippet chunks like [`Self::run_search_stream`].
    /// The counts go out as the first event so the UI can show totals and the
    /// facet tree before the first snippet chunk is even built.
    #[allow(clippy::too_many_arguments)]
    fn run_search_stream_with_counts<F>(
        &self,
        query: Box<dyn Query>,
        make_highlight: F,
        text_field: Field,
        limit: u32,
        offset: u32,
        order: &ResultsOrder,
        hl: &HighlightConfig,
        chunk_size: u32,
        truncated: bool,
        grouping: Option<&ResultGrouping>,
        sink: &StreamSink<SearchStreamUpdate>,
    ) -> Result<()>
    where
        F: FnOnce(&Searcher) -> Result<HighlightPlan>,
    {
        let searcher = self.index_reader.searcher();
        let chunk_size = (chunk_size.max(1)) as usize;

        if let Some(grouping) = grouping {
            // מעבר אינדקס אחד: קיבוץ + ספירות פר-ספר יחד, כמו המסלול השטוח.
            let collectors = (GroupCollector::new(grouping, order), BookCountCollector);
            let ((), book_counts) = searcher.search(&*query, &collectors)?;
            let page = Self::finalize_grouped(collectors.0.take_hits(), limit, offset);
            if sink
                .add(SearchStreamUpdate {
                    total_count: Some(page.raw_total),
                    book_counts: Some(book_counts),
                    results: Vec::new(),
                    truncated: truncated || page.truncated,
                    group_count: Some(page.group_count),
                })
                .is_err()
            {
                return Ok(());
            }
            let results = Self::build_grouped_results(
                &self.schema,
                &searcher,
                &query,
                make_highlight,
                text_field,
                hl,
                &page,
            )?;
            for chunk in results.chunks(chunk_size) {
                if sink
                    .add(SearchStreamUpdate {
                        total_count: None,
                        book_counts: None,
                        results: chunk.to_vec(),
                        truncated: false,
                        group_count: None,
                    })
                    .is_err()
                {
                    break;
                }
            }
            return Ok(());
        }

        let (addresses, total_count, book_counts): (Vec<DocAddress>, u32, HashMap<String, u32>) =
            match order {
                ResultsOrder::Catalogue => {
                    let top_collector = TopDocs::with_limit(limit as usize)
                        .and_offset(offset as usize)
                        .order_by_fast_field::<u64>("id", Order::Asc);
                    let (top_docs, count, by_book) =
                        searcher.search(&*query, &(top_collector, Count, BookCountCollector))?;
                    let addrs = top_docs.into_iter().map(|(_, addr)| addr).collect();
                    (addrs, count as u32, by_book)
                }
                ResultsOrder::Generation => {
                    let top_collector = TopDocs::with_limit(limit as usize)
                        .and_offset(offset as usize)
                        .order_by_fast_field::<u64>("generationSort", Order::Asc);
                    let (top_docs, count, by_book) =
                        searcher.search(&*query, &(top_collector, Count, BookCountCollector))?;
                    let addrs = top_docs.into_iter().map(|(_, addr)| addr).collect();
                    (addrs, count as u32, by_book)
                }
                ResultsOrder::Relevance => {
                    let top_collector = TopDocs::with_limit(limit as usize)
                        .and_offset(offset as usize)
                        .order_by_score();
                    let (top_docs, count, by_book) =
                        searcher.search(&*query, &(top_collector, Count, BookCountCollector))?;
                    let addrs = top_docs.into_iter().map(|(_, addr)| addr).collect();
                    (addrs, count as u32, by_book)
                }
            };

        // Counts first — the page addresses may be empty (offset past the
        // end) while the totals are still meaningful.
        if sink
            .add(SearchStreamUpdate {
                total_count: Some(total_count),
                book_counts: Some(book_counts),
                results: Vec::new(),
                truncated,
                group_count: None,
            })
            .is_err()
        {
            return Ok(());
        }
        if addresses.is_empty() {
            return Ok(());
        }

        let plan = Self::resolve_highlight(&searcher, make_highlight);
        let hl_q: &dyn Query = plan.query.as_deref().unwrap_or(query.as_ref());
        let phrase = plan.phrase.as_ref();
        // One generator for the whole stream: creating it resolves term
        // doc-frequencies, which is too expensive to repeat per chunk.
        let snippet_generator = Self::make_snippet_generator(&searcher, hl_q, text_field, hl)?;
        for chunk in addresses.chunks(chunk_size) {
            let results = Self::build_results_with_generator(
                &self.schema,
                &searcher,
                &snippet_generator,
                text_field,
                chunk.to_vec(),
                hl,
                phrase,
            )?;
            // If the Dart side cancelled the stream, stop early.
            if sink
                .add(SearchStreamUpdate {
                    total_count: None,
                    book_counts: None,
                    results,
                    truncated: false,
                    group_count: None,
                })
                .is_err()
            {
                break;
            }
        }
        Ok(())
    }

    /// Invokes a highlight-query builder against the search's `searcher`,
    /// degrading a build failure to an empty plan (no highlight query, no
    /// phrase filter) instead of failing the whole search. An empty plan makes
    /// the caller fall back to the main query, which already exposes its terms
    /// when it is a Term/Phrase/TermSet query.
    fn resolve_highlight<F>(searcher: &Searcher, make_highlight: F) -> HighlightPlan
    where
        F: FnOnce(&Searcher) -> Result<HighlightPlan>,
    {
        make_highlight(searcher).unwrap_or_else(|_| HighlightPlan::none())
    }

    fn collect_addresses(
        searcher: &Searcher,
        query: &dyn Query,
        limit: u32,
        offset: u32,
        order: &ResultsOrder,
    ) -> Result<Vec<DocAddress>> {
        let addresses = match order {
            ResultsOrder::Catalogue => {
                // and_offset is set on TopDocs before calling order_by_fast_field,
                // which consumes self and preserves the offset configuration.
                let collector = TopDocs::with_limit(limit as usize)
                    .and_offset(offset as usize)
                    .order_by_fast_field::<u64>("id", Order::Asc);
                searcher
                    .search(query, &collector)?
                    .into_iter()
                    .map(|(_, addr)| addr)
                    .collect()
            }
            ResultsOrder::Generation => {
                let collector = TopDocs::with_limit(limit as usize)
                    .and_offset(offset as usize)
                    .order_by_fast_field::<u64>("generationSort", Order::Asc);
                searcher
                    .search(query, &collector)?
                    .into_iter()
                    .map(|(_, addr)| addr)
                    .collect()
            }
            ResultsOrder::Relevance => {
                let collector = TopDocs::with_limit(limit as usize)
                    .and_offset(offset as usize)
                    .order_by_score();
                searcher
                    .search(query, &collector)?
                    .into_iter()
                    .map(|(_, addr)| addr)
                    .collect()
            }
        };
        Ok(addresses)
    }

    /// מריץ את שאילתת החיפוש עם [`GroupCollector`] וגוזר את עמוד הקבוצות
    /// המבוקש (`limit`/`offset` בקבוצות). CPU: מעבר אינדקס מלא אחד — אותה
    /// מחלקת עלות כמו ספירת-כל/ספירה-פר-ספר שהמסך ממילא מריץ. זיכרון:
    /// O(קבוצות), קצוץ ב-[`GROUP_COLLECTOR_MAX_GROUPS`] — בניגוד לספירות,
    /// שמחזיקות מונה-לספר בלבד.
    fn collect_grouped(
        searcher: &Searcher,
        query: &dyn Query,
        grouping: &ResultGrouping,
        limit: u32,
        offset: u32,
        order: &ResultsOrder,
    ) -> Result<GroupedPage> {
        let collector = GroupCollector::new(grouping, order);
        searcher.search(query, &collector)?;
        Ok(Self::finalize_grouped(collector.take_hits(), limit, offset))
    }

    /// ממיין את הקבוצות לפי הנציג הטוב ביותר וגוזר את העמוד המבוקש.
    fn finalize_grouped(hits: GroupedHits, limit: u32, offset: u32) -> GroupedPage {
        let group_count = hits.groups.len() as u32;
        let mut groups: Vec<GroupAcc> = hits.groups.into_values().collect();
        groups.sort_by_key(|g| (g.best.0, g.best.1));
        let reps: Vec<GroupedRep> = groups
            .into_iter()
            .skip(offset as usize)
            .take(limit as usize)
            .map(|g| GroupedRep {
                id: g.best.1,
                address: g.best.2,
                count: g.count,
                siblings: g.siblings.iter().map(|s| s.2).collect(),
            })
            .collect();
        GroupedPage {
            reps,
            group_count,
            raw_total: hits.raw_total,
            truncated: hits.truncated,
        }
    }

    /// בונה את תוצאות העמוד המקובץ: snippet מלא לנציגים (כמו במסלול
    /// השטוח), ולכל נציג — מונה הקבוצה וחברותיה (שליפת doc-store קלה,
    /// בלי snippet). התאמת נציג→קבוצה נעשית לפי `id` (build_results עשוי
    /// לדלג על מסמך שאינו ניתן לקריאה).
    #[allow(clippy::too_many_arguments)]
    fn build_grouped_results<F>(
        schema: &Schema,
        searcher: &Searcher,
        query: &Box<dyn Query>,
        make_highlight: F,
        text_field: Field,
        hl: &HighlightConfig,
        page: &GroupedPage,
    ) -> Result<Vec<SearchResult>>
    where
        F: FnOnce(&Searcher) -> Result<HighlightPlan>,
    {
        if page.reps.is_empty() {
            return Ok(Vec::new());
        }
        let plan = Self::resolve_highlight(searcher, make_highlight);
        let hl_q: &dyn Query = plan.query.as_deref().unwrap_or(query.as_ref());
        let addresses: Vec<DocAddress> = page.reps.iter().map(|r| r.address).collect();
        let mut results = Self::build_results(
            schema,
            searcher,
            hl_q,
            text_field,
            addresses,
            hl,
            plan.phrase.as_ref(),
        )?;
        let rep_by_id: HashMap<u64, &GroupedRep> = page.reps.iter().map(|r| (r.id, r)).collect();
        let fields = SiblingFields::resolve(schema)?;
        for result in &mut results {
            let Some(rep) = rep_by_id.get(&result.id) else {
                continue;
            };
            result.merged_count = rep.count;
            result.merged = rep
                .siblings
                .iter()
                .filter_map(|addr| Self::fetch_merged_sibling(&fields, searcher, *addr).transpose())
                .collect::<Result<Vec<_>>>()?;
        }
        Ok(results)
    }

    /// שליפת doc-store קלה לחברת קבוצה — מיקום בלבד, בלי snippet.
    /// מסמך שאינו ניתן לקריאה נשמט בשקט מהרשימה (המונה כבר ספר אותו).
    fn fetch_merged_sibling(
        fields: &SiblingFields,
        searcher: &Searcher,
        address: DocAddress,
    ) -> Result<Option<MergedSibling>> {
        let doc = match searcher.doc::<TantivyDocument>(address) {
            Ok(d) => d,
            Err(e) => {
                log::error!(
                    "dropping merged sibling: doc store read failed at segment {} doc {}: {e}",
                    address.segment_ord,
                    address.doc_id
                );
                return Ok(None);
            }
        };
        Ok(Some(MergedSibling {
            title: doc
                .get_first(fields.title)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            reference: doc
                .get_first(fields.reference)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            id: doc
                .get_first(fields.id)
                .and_then(|v| v.as_u64())
                .unwrap_or_default(),
            segment: doc
                .get_first(fields.segment)
                .and_then(|v| v.as_u64())
                .unwrap_or_default(),
            is_pdf: doc
                .get_first(fields.is_pdf)
                .and_then(|v| v.as_bool())
                .unwrap_or_default(),
            file_path: doc
                .get_first(fields.file_path)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        }))
    }

    /// Builds display-highlight patterns from the index terms this query
    /// actually matches — the same automaton scan the search and the snippet
    /// highlighter run — so a document found via any variant (typo,
    /// morphological affix, partial word) highlights exactly that variant in
    /// an opened book. Full parity with search by construction: the automatons
    /// are the very `regex_terms` branches `prepare_advanced_query` builds for
    /// the search query.
    ///
    /// A word whose branches match nothing in this index (or fail to compile)
    /// falls back to the query-shape pattern of [`generate_highlight_pattern`],
    /// so the result is never worse than the pure-string one. Runs FST scans
    /// against the term dictionary — async, unlike the sync pure-string
    /// fallback; the app fetches it once per search-parameter change and
    /// caches the compiled `RegExp`s.
    pub fn generate_index_highlight_pattern(
        &self,
        query: String,
        distance: u32,
        custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
    ) -> Result<Option<HighlightPattern>> {
        let advanced = hebrew_query::prepare_advanced_query(
            &query,
            distance,
            &custom_spacing,
            &alternative_words,
            &search_options,
        );
        let searcher = self.index_reader.searcher();

        // Per word: compile the word's branches and collect the index terms
        // they match. `automaton_highlight_terms` splits its budget across the
        // word's branches, so a wide word (partial + typo) still spreads its
        // term allowance over all variants instead of exhausting it on the
        // first branch; the display char budget bounds the final pattern size.
        let mut per_word_terms: Vec<Vec<String>> = Vec::with_capacity(advanced.regex_terms.len());
        for word_pattern in &advanced.regex_terms {
            let automatons: Vec<tantivy_fst::Regex> = word_pattern
                .branches()
                .iter()
                .filter_map(|branch| match tantivy_fst::Regex::new(branch) {
                    Ok(re) => Some(re),
                    Err(e) => {
                        // A branch the search itself would reject: skip it for
                        // highlighting (the search surfaces the error), keep
                        // painting what the remaining branches match.
                        log::error!(
                            "highlight branch failed to compile ({} chars): {e}",
                            branch.chars().count()
                        );
                        None
                    }
                })
                .collect();
            let mut matched: HashSet<String> = if automatons.is_empty() {
                HashSet::new()
            } else {
                self.automaton_highlight_terms(&searcher, &automatons)?
            };
            // Single-word typo path: the search expands typo coverage through
            // Levenshtein-1 automatons, not regex branches — run the same
            // scan here so a document found via any edit-distance-1 variant
            // highlights that variant (search↔highlight parity).
            if !advanced.typo_tokens.is_empty() {
                let builder = LevenshteinAutomatonBuilder::new(1, true);
                let typo_automatons: Vec<DfaWrapper> = advanced
                    .typo_tokens
                    .iter()
                    .map(|t| DfaWrapper(builder.build_dfa(t)))
                    .collect();
                matched.extend(self.automaton_highlight_terms(&searcher, &typo_automatons)?);
            }
            per_word_terms.push(matched.into_iter().collect());
        }

        Ok(display_highlight::build_display_highlight_from_terms(
            &query,
            distance,
            &custom_spacing,
            &alternative_words,
            &search_options,
            &per_word_terms,
        )
        .map(|hl| HighlightPattern {
            combined_pattern: hl.combined_pattern,
            word_patterns: hl.word_patterns,
            word_boundary_eligible: hl.word_boundary_eligible,
        }))
    }

    /// Fuzzy-mode counterpart of [`Self::generate_index_highlight_pattern`]:
    /// paints the index terms within `max_distance` edits of each query token,
    /// plus the dictionary morphological forms when a magic dictionary is
    /// loaded — mirroring [`Self::build_fuzzy_highlight`]'s term collection,
    /// so an opened book highlights exactly what the fuzzy search matched.
    pub fn generate_index_fuzzy_highlight_pattern(
        &self,
        query: String,
        max_distance: u8,
    ) -> Result<Option<HighlightPattern>> {
        anyhow::ensure!(
            max_distance <= 2,
            "fuzzy highlight distance is limited to 2, got {max_distance}"
        );
        let tokens = self.index_token_texts(&query)?;
        // `build_display_highlight_from_terms` walks `split_query_words`;
        // the analyzer tokens are aligned with it by design, but if they ever
        // diverge, feeding misaligned terms would paint the wrong word — fall
        // back to the query-shape pattern instead.
        let words = hebrew_query::split_query_words(&hebrew_query::normalize_for_index(&query));
        let per_word_terms: Vec<Vec<String>> = if tokens.len() != words.len() {
            Vec::new()
        } else {
            let searcher = self.index_reader.searcher();
            let builder = LevenshteinAutomatonBuilder::new(max_distance, true);
            let mut collected = Vec::with_capacity(tokens.len());
            for token in &tokens {
                let mut matched = if max_distance == 0 {
                    HashSet::new()
                } else {
                    self.automaton_highlight_terms(
                        &searcher,
                        &[DfaWrapper(builder.build_dfa(token))],
                    )?
                };
                matched.insert(token.clone());
                if max_distance > 0 {
                    if let Some(dict) = self.magic_dict.as_ref() {
                        for form in dict.highlight_forms(token, MAX_LEXICAL_FORMS) {
                            matched.insert(form);
                        }
                    }
                }
                collected.push(matched.into_iter().collect());
            }
            collected
        };

        Ok(display_highlight::build_display_highlight_from_terms(
            &query,
            0,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &per_word_terms,
        )
        .map(|hl| HighlightPattern {
            combined_pattern: hl.combined_pattern,
            word_patterns: hl.word_patterns,
            word_boundary_eligible: hl.word_boundary_eligible,
        }))
    }

    /// One index-term set per query word (in phrase order), each materialized by
    /// streaming the term dictionary through that word's regex automaton — the
    /// same FST scan the search uses, so the sets contain exactly the
    /// morphological variants (prefixes, suffixes, alternatives) that genuinely
    /// matched. The per-word budget split mirrors
    /// [`Self::automaton_highlight_terms`], so their union equals the merged
    /// highlight term set the flat query paints with.
    fn phrase_per_word_terms(
        &self,
        searcher: &Searcher,
        regex_terms: &[String],
        field: Field,
    ) -> Result<Vec<HashSet<String>>> {
        let cap = (MAX_HIGHLIGHT_TERMS / regex_terms.len().max(1)).max(1);
        regex_terms
            .iter()
            .map(|pattern| {
                let re = tantivy_fst::Regex::new(pattern)
                    .map_err(|e| anyhow::anyhow!("invalid highlight regex {pattern:?}: {e}"))?;
                Ok(self
                    .automaton_terms_in_field(searcher, field, &re, cap)?
                    .into_iter()
                    .collect())
            })
            .collect()
    }

    /// Flattens per-word term sets into the `TermSetQuery` that drives
    /// `SnippetGenerator`'s fragment selection and highlighting — `RegexPhraseQuery`
    /// exposes no static terms of its own. `None` when nothing matched (the
    /// caller then falls back to the main query).
    fn terms_query_from_word_sets(
        &self,
        per_word_terms: &[HashSet<String>],
        text_f: Field,
    ) -> Result<Option<Box<dyn Query>>> {
        let terms: Vec<Term> = per_word_terms
            .iter()
            .flat_map(|set| set.iter())
            .map(|t| Term::from_field_text(text_f, t))
            .collect();
        Ok((!terms.is_empty()).then(|| Box::new(TermSetQuery::new(terms)) as Box<dyn Query>))
    }

    /// Highlight plan for an advanced (regex) search.
    ///
    /// A single regex term runs as a `TermSetQuery` (see
    /// [`Self::single_regex_term_query`]) whose materialized terms ARE the terms
    /// that matched, so the main query already exposes them to `SnippetGenerator`
    /// — no highlight query, no phrase filter. The multi-term `RegexPhraseQuery`
    /// exposes no static terms, so it needs a separately-materialized flat
    /// highlight query AND — being a phrase — the per-word filter that keeps only
    /// in-order, per-pair-within-allowance occurrences painted (parity with the
    /// search's gap verification).
    ///
    /// `gaps` is the query's resolved per-pair allowance vector, which already
    /// folds `custom_spacing` in (else the global `distance` for every pair) —
    /// passing the raw `distance` would reject a spacing-permitted match and
    /// fall back to the broad term highlight.
    /// Scope-aware entry: the word-distance scope keeps the phrase filter
    /// (order + per-pair allowance) of [`Self::advanced_highlight_plan`];
    /// the paragraph/section scopes impose no order or distance inside a
    /// line, so every occurrence of every word variant is a true match to
    /// paint — flat term-union highlighting, no phrase filter.
    /// `acronym_alts` — חלופות פענוח ר"ת (תבניות-ליטרל פר-מילה): מילות כל
    /// חלופה מתממשות ומצטרפות לאיחוד ההדגשה השטוח, כך שמסמך שנמצא דרך
    /// החלופה ייצבע (דרך נפילת מסנן-הביטוי לצביעת-הטרמים הרחבה).
    fn advanced_highlight_plan_for_scope(
        &self,
        searcher: &Searcher,
        regex_terms: &[String],
        gaps: &[u32],
        voc: &VocalizedFlags,
        scope: &SearchScope,
        match_mode: &WordMatch,
        acronym_alts: &[Vec<String>],
    ) -> Result<HighlightPlan> {
        // בהתאמה חלקית אין דרישת סדר/מרחק — מסנן הביטוי של מסלול המרחק
        // היה מוחק הדגשות לגיטימיות של מילים בודדות או בסדר הפוך.
        let partial = !matches!(match_mode, WordMatch::All);
        match scope {
            SearchScope::WordDistance if !partial => {
                self.advanced_highlight_plan(searcher, regex_terms, gaps, voc, acronym_alts)
            }
            _ => {
                if regex_terms.len() < 2 && acronym_alts.is_empty() {
                    return Ok(HighlightPlan::none());
                }
                let field = self.search_text_field(voc)?;
                let mut word_sets = self.phrase_per_word_terms(searcher, regex_terms, field)?;
                for alt in acronym_alts {
                    word_sets.extend(self.phrase_per_word_terms(searcher, alt, field)?);
                }
                let query = self.terms_query_from_word_sets(&word_sets, field)?;
                Ok(HighlightPlan {
                    query,
                    phrase: None,
                })
            }
        }
    }

    fn advanced_highlight_plan(
        &self,
        searcher: &Searcher,
        regex_terms: &[String],
        gaps: &[u32],
        voc: &VocalizedFlags,
        acronym_alts: &[Vec<String>],
    ) -> Result<HighlightPlan> {
        if regex_terms.len() < 2 && acronym_alts.is_empty() {
            return Ok(HighlightPlan::none());
        }
        let field = self.search_text_field(voc)?;
        let analyzer = if voc.any() {
            "hebrew_vocalized"
        } else {
            "hebrew"
        };
        let per_word_terms = self.phrase_per_word_terms(searcher, regex_terms, field)?;
        // איחוד ההדגשה השטוח נושא גם את מילות חלופות הר"ת; מסנן-הביטוי
        // נשאר של השאילתה הראשית בלבד (רק במסלול הרב-מילי) — פרגמנט
        // שנמצא דרך חלופה נופל לצביעה הרחבה וכל מילות החלופה נצבעות.
        let mut all_word_sets = per_word_terms.clone();
        for alt in acronym_alts {
            all_word_sets.extend(self.phrase_per_word_terms(searcher, alt, field)?);
        }
        let query = self.terms_query_from_word_sets(&all_word_sets, field)?;
        let phrase = (per_word_terms.len() >= 2).then(|| PhraseHighlight {
            per_word_terms,
            gaps: gaps.to_vec(),
            analyzer,
        });
        Ok(HighlightPlan { query, phrase })
    }

    /// Highlight plan for an exact (`Term`/`PhraseQuery`) search. A single term
    /// needs nothing — the `TermQuery` highlights itself. A multi-word
    /// `PhraseQuery` already exposes its terms to `SnippetGenerator`, so it needs
    /// no separate highlight query, but it is a strict-adjacency phrase, so it
    /// gets an all-zero gaps filter that drops every non-adjacent occurrence.
    ///
    /// The vocalized arm mirrors the advanced plan instead: each token's
    /// required-marks pattern is materialized against the vocalized
    /// dictionary (the `RegexPhraseQuery` exposes no static terms), and the
    /// phrase filter re-tokenizes fragments with the vocalized analyzer.
    fn exact_highlight_plan(
        &self,
        searcher: &Searcher,
        query_str: &str,
        voc: &VocalizedFlags,
    ) -> Result<HighlightPlan> {
        if voc.any() {
            let tokens = self.index_token_texts_with("hebrew_vocalized_query", query_str)?;
            if tokens.len() < 2 {
                return Ok(HighlightPlan::none());
            }
            let patterns: Vec<String> = tokens
                .iter()
                .map(|t| hebrew_query::vocalized_token_pattern(t, voc))
                .collect();
            let field = self.schema.get_field("textVocalized")?;
            let per_word_terms = self.phrase_per_word_terms(searcher, &patterns, field)?;
            let query = self.terms_query_from_word_sets(&per_word_terms, field)?;
            let gaps = vec![0; per_word_terms.len().saturating_sub(1)];
            return Ok(HighlightPlan {
                query,
                phrase: Some(PhraseHighlight {
                    per_word_terms,
                    gaps,
                    analyzer: "hebrew_vocalized",
                }),
            });
        }
        let tokens = self.index_token_texts(query_str)?;
        if tokens.len() < 2 {
            return Ok(HighlightPlan::none());
        }
        let per_word_terms: Vec<HashSet<String>> =
            tokens.into_iter().map(|t| HashSet::from([t])).collect();
        let gaps = vec![0; per_word_terms.len().saturating_sub(1)];
        Ok(HighlightPlan {
            query: None,
            phrase: Some(PhraseHighlight {
                per_word_terms,
                gaps,
                analyzer: "hebrew",
            }),
        })
    }

    /// Highlight plan for the approximate (`fuzzy`) search. Always builds the
    /// flat highlight query (fuzzy/lexical automatons expose no static terms).
    /// Adds a phrase filter only for the lexical multi-word path — the sole
    /// fuzzy path that builds a `RegexPhraseQuery`. Plain fuzzy multi-word is a
    /// per-token AND, where every occurrence of every word is a real hit and
    /// must stay highlighted, so it carries no filter.
    fn fuzzy_highlight_plan(
        &self,
        searcher: &Searcher,
        term_texts: &[String],
        max_distance: u8,
    ) -> Result<HighlightPlan> {
        let query = self
            .build_fuzzy_highlight(searcher, term_texts, max_distance)
            .ok();
        let phrase = if term_texts.len() >= 2 && self.magic_dict.is_some() && max_distance > 0 {
            let per_word_terms =
                self.lexical_phrase_per_word_terms(searcher, term_texts, max_distance)?;
            let gaps = vec![LEXICAL_FUZZY_PHRASE_SLOP; per_word_terms.len().saturating_sub(1)];
            Some(PhraseHighlight {
                per_word_terms,
                gaps,
                analyzer: "hebrew",
            })
        } else {
            None
        };
        Ok(HighlightPlan { query, phrase })
    }

    /// Per-word term sets for the lexical fuzzy *phrase* path: each word's
    /// edit-distance matches plus its blacklist-filtered dictionary forms and
    /// quote-free spelling — mirroring
    /// [`Self::generate_index_fuzzy_highlight_pattern`] so the results snippet
    /// and an opened book paint the same fuzzy variants.
    fn lexical_phrase_per_word_terms(
        &self,
        searcher: &Searcher,
        tokens: &[String],
        max_distance: u8,
    ) -> Result<Vec<HashSet<String>>> {
        let builder = LevenshteinAutomatonBuilder::new(max_distance, true);
        tokens
            .iter()
            .map(|token| {
                let mut matched = self
                    .automaton_highlight_terms(searcher, &[DfaWrapper(builder.build_dfa(token))])?;
                matched.insert(token.clone());
                if let Some(clean) = Self::quoteless_variant(token) {
                    matched.insert(clean);
                }
                if let Some(dict) = self.magic_dict.as_ref() {
                    for form in dict.highlight_forms(token, MAX_LEXICAL_FORMS) {
                        matched.insert(form);
                    }
                }
                Ok(matched)
            })
            .collect()
    }

    /// Fuzzy-mode counterpart of [`Self::phrase_per_word_terms`]'s scan:
    /// materializes the `text` terms each query term matches within
    /// `max_distance` edits. `FuzzyTermQuery` is automaton-based like the regex
    /// queries and exposes no static terms to `SnippetGenerator`, so without
    /// this fuzzy results would render with no highlighting.
    fn build_fuzzy_highlight_query(
        &self,
        searcher: &Searcher,
        term_texts: &[String],
        max_distance: u8,
    ) -> Result<Box<dyn Query>> {
        // FuzzyTermQuery itself rejects distances above 2.
        anyhow::ensure!(
            max_distance <= 2,
            "fuzzy highlight distance is limited to 2, got {max_distance}"
        );
        // Same builder configuration as the search's FuzzyTermQuery
        // (transposition counts as one edit), so the highlighted terms are
        // exactly the terms the query can match.
        let builder = LevenshteinAutomatonBuilder::new(max_distance, true);
        let automatons: Vec<DfaWrapper> = term_texts
            .iter()
            .map(|t| DfaWrapper(builder.build_dfa(t)))
            .collect();
        self.build_automaton_highlight_query(searcher, &automatons)
    }

    /// Highlight query for the approximate (`fuzzy`) path, branching on whether
    /// a `MagicDictionary` is loaded — the highlight terms must mirror whatever
    /// [`Self::build_fuzzy_search_query`] matched. Takes the search's own
    /// `searcher` so highlight terms come from the same index snapshot.
    fn build_fuzzy_highlight(
        &self,
        searcher: &Searcher,
        term_texts: &[String],
        max_distance: u8,
    ) -> Result<Box<dyn Query>> {
        if self.magic_dict.is_some() && max_distance > 0 {
            self.build_lexical_fuzzy_highlight_query(searcher, term_texts, max_distance)
        } else {
            self.build_fuzzy_highlight_query(searcher, term_texts, max_distance)
        }
    }

    /// Like [`Self::build_fuzzy_highlight_query`] but also paints the lexical
    /// forms injected by [`Self::build_lexical_fuzzy_query`]. The blacklist is
    /// applied here (highlight only): hallucinated lemmas still expanded recall
    /// but are not highlighted.
    fn build_lexical_fuzzy_highlight_query(
        &self,
        searcher: &Searcher,
        term_texts: &[String],
        max_distance: u8,
    ) -> Result<Box<dyn Query>> {
        anyhow::ensure!(
            max_distance <= 2,
            "fuzzy highlight distance is limited to 2, got {max_distance}"
        );
        let dict = self
            .magic_dict
            .as_ref()
            .context("lexical fuzzy highlight requires a loaded magic dictionary")?;
        let text_f = self.schema.get_field("text")?;

        // Start from the edit-distance terms (same automatons as search)...
        let builder = LevenshteinAutomatonBuilder::new(max_distance, true);
        let automatons: Vec<DfaWrapper> = term_texts
            .iter()
            .map(|t| DfaWrapper(builder.build_dfa(t)))
            .collect();
        let mut matched = self.automaton_highlight_terms(searcher, &automatons)?;

        // ...then add the literal tokens and the (blacklist-filtered) lexical
        // forms per token. The exact token can otherwise be omitted when a broad
        // fuzzy automaton exhausts its highlight-term budget first.
        for token in term_texts {
            matched.insert(token.clone());
            for form in dict.highlight_forms(token, MAX_LEXICAL_FORMS) {
                matched.insert(form);
            }
        }

        let terms: Vec<Term> = matched
            .into_iter()
            .map(|t| Term::from_field_text(text_f, &t))
            .collect();
        Ok(Box::new(TermSetQuery::new(terms)))
    }

    fn build_automaton_highlight_query<A>(
        &self,
        searcher: &Searcher,
        automatons: &[A],
    ) -> Result<Box<dyn Query>>
    where
        A: Automaton,
        A::State: Clone,
    {
        let text_f = self.schema.get_field("text")?;
        let matched = self.automaton_highlight_terms(searcher, automatons)?;
        let terms: Vec<Term> = matched
            .into_iter()
            .map(|t| Term::from_field_text(text_f, &t))
            .collect();
        Ok(Box::new(TermSetQuery::new(terms)))
    }

    /// Collects the distinct `text`-index terms the given automatons match,
    /// bounded by [`MAX_HIGHLIGHT_TERMS`] split evenly across automatons.
    /// Shared by the regex, fuzzy, and lexical-fuzzy highlight builders.
    fn automaton_highlight_terms<A>(
        &self,
        searcher: &Searcher,
        automatons: &[A],
    ) -> Result<HashSet<String>>
    where
        A: Automaton,
        A::State: Clone,
    {
        let mut matched: HashSet<String> = HashSet::new();
        // Split the term budget evenly between automatons: a global cap would
        // let one broad first word exhaust it and leave the remaining query
        // words with no highlighting at all.
        let per_automaton_cap = (MAX_HIGHLIGHT_TERMS / automatons.len().max(1)).max(1);
        for automaton in automatons {
            for term in self.automaton_terms(searcher, automaton, per_automaton_cap)? {
                matched.insert(term);
            }
        }
        Ok(matched)
    }

    fn automaton_terms<A>(
        &self,
        searcher: &Searcher,
        automaton: &A,
        cap: usize,
    ) -> Result<Vec<String>>
    where
        A: Automaton,
        A::State: Clone,
    {
        let text_f = self.schema.get_field("text")?;
        self.automaton_terms_in_field(searcher, text_f, automaton, cap)
    }

    /// [`Self::automaton_terms`] against an explicit field's dictionary.
    fn automaton_terms_in_field<A>(
        &self,
        searcher: &Searcher,
        text_f: Field,
        automaton: &A,
        cap: usize,
    ) -> Result<Vec<String>>
    where
        A: Automaton,
        A::State: Clone,
    {
        let mut matched = Vec::new();
        let mut seen = HashSet::new();
        'segments: for reader in searcher.segment_readers() {
            let inverted = reader.inverted_index(text_f)?;
            let mut stream = inverted.terms().search(automaton).into_stream()?;
            while stream.advance() {
                if let Ok(term) = std::str::from_utf8(stream.key()) {
                    // contains-before-insert: a term already seen in an
                    // earlier segment costs no allocation at all.
                    if !seen.contains(term) {
                        seen.insert(term.to_string());
                        matched.push(term.to_string());
                        if matched.len() >= cap {
                            break 'segments;
                        }
                    }
                }
            }
        }
        Ok(matched)
    }

    /// Creates a `SnippetGenerator` for `text_field` (the plain `text` field,
    /// or `textVocalized` on the vocalized paths), configured from `hl`.
    /// Creation resolves the doc-frequencies of every query term, so reuse one
    /// generator across chunks instead of recreating it per call.
    fn make_snippet_generator(
        searcher: &Searcher,
        query: &dyn Query,
        text_field: Field,
        hl: &HighlightConfig,
    ) -> Result<SnippetGenerator> {
        let mut snippet_generator = SnippetGenerator::create(searcher, query, text_field)?;
        snippet_generator.set_max_num_chars(hl.max_chars as usize);
        Ok(snippet_generator)
    }

    /// Re-derives a snippet's highlight markup so only complete, in-order
    /// phrase occurrences stay painted (see [`PhraseHighlight`]).
    ///
    /// Tokenizes the fragment with the same `text`-field analyzer the index and
    /// `SnippetGenerator` use, tags each token with the query words it can fill,
    /// then keeps the byte ranges of tokens forming an occurrence
    /// `w0 … w1 … w_{k-1}` where each adjacent pair `w-1, w` is at most
    /// `gaps[w-1]` intermediate tokens apart — the greedy, leftmost,
    /// non-overlapping match `display_highlight`'s combined pattern performs
    /// in an opened book.
    ///
    /// Returns `None` when the fragment holds no complete occurrence, so the
    /// caller falls back to the plain term highlight instead of painting
    /// nothing (never less context than before).
    fn phrase_filtered_snippet_html(
        searcher: &Searcher,
        fragment: &str,
        phrase: &PhraseHighlight,
        hl: &HighlightConfig,
    ) -> Option<String> {
        let word_count = phrase.per_word_terms.len();
        if word_count < 2 {
            return None;
        }
        let mut analyzer = searcher.index().tokenizers().get(phrase.analyzer)?;

        // Candidate = a fragment token that can fill at least one query word.
        // `order` is the tokenizer's `position` — one increment per *word*:
        // the quote-free twin token an indexing analyzer emits (ראו
        // `emit_quote_free`) shares its word's position, so it must not
        // inflate the intermediate-word gap `order_b - order_a - 1`.
        struct Candidate {
            order: usize,
            from: usize,
            to: usize,
            words: Vec<usize>,
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut stream = analyzer.token_stream(fragment);
        while let Some(token) = stream.next() {
            let words: Vec<usize> = phrase
                .per_word_terms
                .iter()
                .enumerate()
                .filter_map(|(w, set)| set.contains(token.text.as_str()).then_some(w))
                .collect();
            if !words.is_empty() {
                candidates.push(Candidate {
                    order: token.position,
                    from: token.offset_from,
                    to: token.offset_to,
                    words,
                });
            }
        }

        // Greedy leftmost, non-overlapping scan.
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut ci = 0usize;
        while ci < candidates.len() {
            if candidates[ci].words.contains(&0) {
                let mut chosen = vec![ci];
                let mut cur = ci;
                let mut ok = true;
                for w in 1..word_count {
                    let max_gap = phrase.gaps.get(w - 1).copied().unwrap_or(0) as usize;
                    let mut m = cur + 1;
                    let mut found = None;
                    while m < candidates.len() {
                        // The gap grows monotonically with m, so once it exceeds
                        // the allowance no later candidate can match this word.
                        // saturating: טוקן-תאום חולק עמדה עם מילתו (אין הפרש).
                        if candidates[m]
                            .order
                            .saturating_sub(candidates[cur].order + 1)
                            > max_gap
                        {
                            break;
                        }
                        if candidates[m].words.contains(&w) {
                            found = Some(m);
                            break;
                        }
                        m += 1;
                    }
                    match found {
                        Some(m) => {
                            chosen.push(m);
                            cur = m;
                        }
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    for &c in &chosen {
                        ranges.push((candidates[c].from, candidates[c].to));
                    }
                    ci = cur + 1;
                    continue;
                }
            }
            ci += 1;
        }

        if ranges.is_empty() {
            return None;
        }

        // Same escaping as tantivy's `Snippet::to_html`. Ranges are built in
        // increasing, non-overlapping order; the guard is defensive.
        ranges.sort_by_key(|&(s, _)| s);
        let mut html = String::new();
        let mut start_from = 0usize;
        for (s, e) in ranges {
            if s < start_from {
                continue;
            }
            html.push_str(&htmlescape::encode_minimal(&fragment[start_from..s]));
            html.push_str(&hl.highlight_prefix);
            html.push_str(&htmlescape::encode_minimal(&fragment[s..e]));
            html.push_str(&hl.highlight_postfix);
            start_from = e;
        }
        html.push_str(&htmlescape::encode_minimal(&fragment[start_from..]));
        Some(html)
    }

    fn build_results(
        schema: &Schema,
        searcher: &Searcher,
        query: &dyn Query,
        text_field: Field,
        addresses: Vec<DocAddress>,
        hl: &HighlightConfig,
        phrase: Option<&PhraseHighlight>,
    ) -> Result<Vec<SearchResult>> {
        let snippet_generator = Self::make_snippet_generator(searcher, query, text_field, hl)?;
        Self::build_results_with_generator(
            schema,
            searcher,
            &snippet_generator,
            text_field,
            addresses,
            hl,
            phrase,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_results_with_generator(
        schema: &Schema,
        searcher: &Searcher,
        snippet_generator: &SnippetGenerator,
        text_field: Field,
        addresses: Vec<DocAddress>,
        hl: &HighlightConfig,
        phrase: Option<&PhraseHighlight>,
    ) -> Result<Vec<SearchResult>> {
        let title_field = schema.get_field("title")?;
        let reference_field = schema.get_field("reference")?;
        // המסלול המנוקד מציג את העותק המנוקד השמור; שדה `text` הרגיל נשאר
        // fallback הגנתי (לא אמור לקרות — כל מסמך שנמצא דרך השדה המנוקד
        // נכתב עם עותק שמור).
        let plain_text_field = schema.get_field("text")?;
        let id_field = schema.get_field("id")?;
        let segment_field = schema.get_field("segment")?;
        let is_pdf_field = schema.get_field("isPdf")?;
        let file_path_field = schema.get_field("filePath")?;

        let mut results = Vec::with_capacity(addresses.len());
        for doc_address in addresses {
            let retrieved_doc = match searcher.doc::<TantivyDocument>(doc_address) {
                Ok(d) => d,
                Err(e) => {
                    // A hit the collectors counted but the doc store cannot
                    // materialize (e.g. a corrupt store block). Skipping it
                    // silently is what shows up in the UI as "3/4 results"
                    // with a load-more button that never delivers — leave a
                    // trace so the mismatch is diagnosable.
                    log::error!(
                        "dropping counted hit: doc store read failed at segment {} doc {}: {e}",
                        doc_address.segment_ord,
                        doc_address.doc_id
                    );
                    continue;
                }
            };

            let title = retrieved_doc
                .get_first(title_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let reference = retrieved_doc
                .get_first(reference_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let text = retrieved_doc
                .get_first(text_field)
                .and_then(|v| v.as_str())
                .or_else(|| {
                    retrieved_doc
                        .get_first(plain_text_field)
                        .and_then(|v| v.as_str())
                })
                .unwrap_or_default()
                .to_string();
            let id = retrieved_doc
                .get_first(id_field)
                .and_then(|v| v.as_u64())
                .unwrap_or_default();
            let segment = retrieved_doc
                .get_first(segment_field)
                .and_then(|v| v.as_u64())
                .unwrap_or_default();
            let is_pdf = retrieved_doc
                .get_first(is_pdf_field)
                .and_then(|v| v.as_bool())
                .unwrap_or_default();
            let file_path = retrieved_doc
                .get_first(file_path_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let mut snippet = snippet_generator.snippet(&text);
            snippet.set_snippet_prefix_postfix(&hl.highlight_prefix, &hl.highlight_postfix);
            // For multi-word phrase queries, tantivy's term-based highlighter
            // paints every occurrence of every query term; re-derive the
            // highlights so only in-order, within-gap phrase occurrences stay
            // painted. Falls back to the plain term highlight when the chosen
            // fragment holds no complete phrase occurrence (never paints less
            // context than before).
            let snippet_html = match phrase {
                Some(pf) => {
                    Self::phrase_filtered_snippet_html(searcher, snippet.fragment(), pf, hl)
                        .unwrap_or_else(|| snippet.to_html())
                }
                None => snippet.to_html(),
            };
            let result_text = if snippet_html.is_empty() {
                text
            } else {
                snippet_html
            };

            results.push(SearchResult {
                title,
                reference,
                text: result_text,
                id,
                segment,
                is_pdf,
                file_path,
                merged_count: 1,
                merged: Vec::new(),
            });
        }
        Ok(results)
    }
}

impl HighlightConfig {
    fn default() -> Self {
        HighlightConfig {
            highlight_prefix: "<font color=red>".to_string(),
            highlight_postfix: "</font>".to_string(),
            max_chars: 800,
        }
    }
}

// ── DfaWrapper ─────────────────────────────────────────────────────────────────

/// Adapts a Levenshtein [`DFA`] to the [`tantivy_fst::Automaton`] trait so the
/// term dictionary can be streamed with the same automaton `FuzzyTermQuery`
/// matches with (mirrors tantivy's internal fuzzy-query wrapper, which is
/// private).
struct DfaWrapper(DFA);

impl Automaton for DfaWrapper {
    type State = u32;

    fn start(&self) -> Self::State {
        self.0.initial_state()
    }

    fn is_match(&self, state: &Self::State) -> bool {
        match self.0.distance(*state) {
            Distance::Exact(_) => true,
            Distance::AtLeast(_) => false,
        }
    }

    fn can_match(&self, state: &Self::State) -> bool {
        *state != SINK_STATE
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        self.0.transition(*state, byte)
    }
}

// ── BookCountCollector ─────────────────────────────────────────────────────────

/// Counts matching documents grouped by `filePath` fast field.
/// Per-segment counts use term ordinals; strings are decoded only in harvest().
struct BookCountCollector;

struct BookCountSegmentCollector {
    str_col: Option<tantivy::columnar::StrColumn>,
    counts: HashMap<u64, u32>,
}

impl Collector for BookCountCollector {
    type Fruit = HashMap<String, u32>;
    type Child = BookCountSegmentCollector;

    fn for_segment(
        &self,
        _seg_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> tantivy::Result<BookCountSegmentCollector> {
        let str_col = reader.fast_fields().str("filePath")?;
        Ok(BookCountSegmentCollector {
            str_col,
            counts: HashMap::new(),
        })
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(
        &self,
        per_segment: Vec<tantivy::Result<HashMap<String, u32>>>,
    ) -> tantivy::Result<HashMap<String, u32>> {
        let mut merged: HashMap<String, u32> = HashMap::new();
        for seg_result in per_segment {
            for (path, count) in seg_result? {
                *merged.entry(path).or_insert(0) += count;
            }
        }
        Ok(merged)
    }
}

impl SegmentCollector for BookCountSegmentCollector {
    type Fruit = tantivy::Result<HashMap<String, u32>>;

    fn collect(&mut self, doc_id: DocId, _score: Score) {
        if let Some(col) = &self.str_col {
            if let Some(term_ord) = col.term_ords(doc_id).next() {
                *self.counts.entry(term_ord).or_insert(0) += 1;
            }
        }
    }

    fn harvest(self) -> tantivy::Result<HashMap<String, u32>> {
        let Some(col) = self.str_col else {
            return Ok(HashMap::new());
        };
        let mut result = HashMap::with_capacity(self.counts.len());
        let mut buf = String::new();
        for (term_ord, count) in self.counts {
            buf.clear();
            if col.ord_to_str(term_ord, &mut buf)? {
                result.insert(buf.clone(), count);
            }
        }
        Ok(result)
    }
}

// ── GroupCollector ─────────────────────────────────────────────────────────────

/// מפתח המיון של מסמך בקיבוץ, כ-u64 עולה: `id` (קטלוג), `generationSort`
/// (דורות), או היפוך ביטי הציון (רלוונטיות — ציון BM25 אי-שלילי, כך
/// שהיפוך הביטים הופך "ציון גבוה" ל"ערך נמוך" והמיון נשאר עולה).
#[derive(Clone, Copy)]
enum GroupSort {
    IdAsc,
    GenerationAsc,
    ScoreDesc,
}

/// `(sort_val, id, address)` — ה-id ייחודי גלובלית ומשמש שובר-שוויון
/// דטרמיניסטי (בציוני רלוונטיות שוויון שכיח).
type GroupEntry = (u64, u64, DocAddress);

/// צבירת קבוצה אחת: מונה, הנציג הטוב ביותר, ועד [`MERGED_SIBLINGS_CAP`]
/// חברות נוספות ממוינות לפי אותו סדר.
struct GroupAcc {
    count: u32,
    best: GroupEntry,
    siblings: Vec<GroupEntry>,
}

impl GroupAcc {
    fn new(entry: GroupEntry) -> Self {
        Self {
            count: 1,
            best: entry,
            siblings: Vec::new(),
        }
    }

    fn push(&mut self, mut entry: GroupEntry) {
        self.count += 1;
        if (entry.0, entry.1) < (self.best.0, self.best.1) {
            std::mem::swap(&mut self.best, &mut entry);
        }
        Self::insert_capped(&mut self.siblings, entry);
    }

    fn insert_capped(siblings: &mut Vec<GroupEntry>, entry: GroupEntry) {
        let pos = siblings.partition_point(|s| (s.0, s.1) <= (entry.0, entry.1));
        if pos < MERGED_SIBLINGS_CAP {
            siblings.insert(pos, entry);
            siblings.truncate(MERGED_SIBLINGS_CAP);
        }
    }
}

/// מפת קבוצות חסומת-[`GROUP_COLLECTOR_MAX_GROUPS`] שמפנה בתקרה את הקבוצה
/// *הגרועה* (לפי הנציג הטוב ביותר) לטובת קבוצה טובה ממנה — כך שהקבוצות
/// הטובות ביותר שורדות בכל סדר סריקה/מיזוג, והעמוד המוחזר מדויק. השארית
/// לאחר תקרה: `group_count` הוא תחתית, ומונה של קבוצה שפונתה וחזרה מאבד
/// את חבריה המוקדמים — שניהם תחת אותו דגל `truncated`.
struct BoundedGroups {
    groups: HashMap<(u8, u64), GroupAcc>,
    /// אינדקס המיון: `(sort_val, id)` של הנציג הטוב + מפתח הקבוצה.
    by_best: std::collections::BTreeSet<(u64, u64, (u8, u64))>,
    truncated: bool,
}

impl BoundedGroups {
    fn new() -> Self {
        Self {
            groups: HashMap::new(),
            by_best: std::collections::BTreeSet::new(),
            truncated: false,
        }
    }

    /// מסמך בודד (מסלול האיסוף פר-סגמנט).
    fn add_entry(&mut self, key: (u8, u64), entry: GroupEntry) {
        if let Some(acc) = self.groups.get_mut(&key) {
            let old_best = (acc.best.0, acc.best.1, key);
            acc.push(entry);
            let new_best = (acc.best.0, acc.best.1, key);
            if new_best != old_best {
                self.by_best.remove(&old_best);
                self.by_best.insert(new_best);
            }
            return;
        }
        self.insert_group(key, GroupAcc::new(entry));
    }

    fn insert_group(&mut self, key: (u8, u64), acc: GroupAcc) {
        let best = (acc.best.0, acc.best.1, key);
        if self.groups.len() >= GROUP_COLLECTOR_MAX_GROUPS {
            self.truncated = true;
            let worst = *self
                .by_best
                .last()
                .expect("cap > 0, so a full map has a worst group");
            if best >= worst {
                return;
            }
            self.by_best.remove(&worst);
            self.groups.remove(&worst.2);
        }
        self.by_best.insert(best);
        self.groups.insert(key, acc);
    }
}

/// פרי החיפוש המקובץ: ספירת התוצאות הגולמית + מפת הקבוצות. מפתח קבוצה
/// הוא `(תג, ערך)` — תג 0 = `sectionId`, תג 1 = `lineHash`, תג 2 =
/// מסמך-יחיד (שורה ללא חתימת דה-דופ במצב `IdenticalText` לעולם אינה
/// מתאחדת, וה-id שלה משמש כמפתח).
///
/// `truncated` — [`GROUP_COLLECTOR_MAX_GROUPS`] נפגעה וקבוצות גרועות
/// פונו (ראו [`BoundedGroups`]); `raw_total` נשאר מדויק.
struct GroupedHits {
    raw_total: u32,
    groups: HashMap<(u8, u64), GroupAcc>,
    truncated: bool,
}

/// נציג קבוצה בעמוד סופי, אחרי מיון וחיתוך `limit`/`offset`.
struct GroupedRep {
    id: u64,
    address: DocAddress,
    count: u32,
    siblings: Vec<DocAddress>,
}

/// עמוד קבוצות סופי — נציגים + ספירות. `truncated` כב-[`GroupedHits`]:
/// תקרת הקבוצות נפגעה, `group_count` הוא תחתית ולא הערך המלא.
struct GroupedPage {
    reps: Vec<GroupedRep>,
    group_count: u32,
    raw_total: u32,
    truncated: bool,
}

/// שדות ה-doc-store שחברת קבוצה נושאת ([`MergedSibling`]) — נפתרים מהסכימה
/// פעם אחת לעמוד ולא פר-sibling.
struct SiblingFields {
    title: Field,
    reference: Field,
    id: Field,
    segment: Field,
    is_pdf: Field,
    file_path: Field,
}

impl SiblingFields {
    fn resolve(schema: &Schema) -> Result<Self> {
        Ok(Self {
            title: schema.get_field("title")?,
            reference: schema.get_field("reference")?,
            id: schema.get_field("id")?,
            segment: schema.get_field("segment")?,
            is_pdf: schema.get_field("isPdf")?,
            file_path: schema.get_field("filePath")?,
        })
    }
}

/// גודל ה-buffer הפר-סגמנטי שמרווח את הנעילה על הצבירה המשותפת —
/// נעילה אחת לכל ~4K מסמכים תואמים במקום נעילה פר-מסמך.
const GROUP_FLUSH_BUFFER: usize = 4_096;

/// הצבירה המשותפת של החיפוש המקובץ: מפה חסומה **אחת לכל החיפוש**.
/// צבירה פר-סגמנט (עם מיזוג ב-merge_fruits) הייתה מחזיקה עד
/// [`GROUP_COLLECTOR_MAX_GROUPS`] קבוצות לכל סגמנט בו-זמנית — עשרות
/// סגמנטים (אינדקס לא ממוזג) היו מצטברים למאות MB לפני המיזוג.
struct SharedGroupedAcc {
    raw_total: u32,
    groups: BoundedGroups,
}

/// אוסף את כל המסמכים התואמים לקבוצות (ראו [`ResultGrouping`]) במעבר
/// אחד, עם נציג-מיטבי וחברות קצוצות-תקרה לכל קבוצה. הזיכרון פרופורציונלי
/// למספר הקבוצות — וזה *אינו* חסום בתקציב ה-postings: המסלולים exact
/// הלא-מנוקד (TermQuery/PhraseQuery) ו-fuzzy (תקציבי האוטומטון חוסמים
/// מספר מונחים, לא מספר מסמכים תואמים) יכולים לעבור על מיליוני שורות.
/// לכן מספר הקבוצות קצוץ גלובלית ב-[`GROUP_COLLECTOR_MAX_GROUPS`] עם
/// degrade מדווח, לא צבירה בלתי-חסומה. התוצאה נשלפת ב-[`Self::take_hits`]
/// אחרי `searcher.search` (ה-Fruit ריק).
struct GroupCollector {
    mode: ResultGrouping,
    sort: GroupSort,
    shared: Arc<Mutex<SharedGroupedAcc>>,
}

impl GroupCollector {
    fn new(grouping: &ResultGrouping, order: &ResultsOrder) -> Self {
        Self {
            mode: *grouping,
            sort: match order {
                ResultsOrder::Catalogue => GroupSort::IdAsc,
                ResultsOrder::Generation => GroupSort::GenerationAsc,
                ResultsOrder::Relevance => GroupSort::ScoreDesc,
            },
            shared: Arc::new(Mutex::new(SharedGroupedAcc {
                raw_total: 0,
                groups: BoundedGroups::new(),
            })),
        }
    }

    /// שולף את התוצאה שנצברה — לקריאה פעם אחת, אחרי `searcher.search`.
    fn take_hits(&self) -> GroupedHits {
        let mut acc = self.shared.lock().expect("group accumulator poisoned");
        GroupedHits {
            raw_total: acc.raw_total,
            truncated: acc.groups.truncated,
            groups: std::mem::take(&mut acc.groups.groups),
        }
    }
}

struct GroupSegmentCollector {
    seg_ord: SegmentOrdinal,
    mode: ResultGrouping,
    sort: GroupSort,
    id_col: tantivy::columnar::Column<u64>,
    /// עמודת המיון (`id`/`generationSort`); `None` במיון לפי ציון.
    sort_col: Option<tantivy::columnar::Column<u64>>,
    /// עמודת מפתח הקבוצה: `sectionId` או `lineHash` לפי המצב.
    key_col: tantivy::columnar::Column<u64>,
    shared: Arc<Mutex<SharedGroupedAcc>>,
    buffer: Vec<((u8, u64), GroupEntry)>,
}

impl GroupSegmentCollector {
    fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let mut acc = self.shared.lock().expect("group accumulator poisoned");
        acc.raw_total = acc.raw_total.saturating_add(self.buffer.len() as u32);
        for (key, entry) in self.buffer.drain(..) {
            acc.groups.add_entry(key, entry);
        }
    }
}

impl Collector for GroupCollector {
    type Fruit = ();
    type Child = GroupSegmentCollector;

    fn for_segment(
        &self,
        seg_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> tantivy::Result<GroupSegmentCollector> {
        let fast = reader.fast_fields();
        let id_col = fast.u64("id")?;
        let sort_col = match self.sort {
            GroupSort::IdAsc => None, // עמודת ה-id כבר בידינו
            GroupSort::GenerationAsc => Some(fast.u64("generationSort")?),
            GroupSort::ScoreDesc => None,
        };
        let key_col = match self.mode {
            ResultGrouping::SameSection => fast.u64("sectionId")?,
            ResultGrouping::IdenticalText => fast.u64("lineHash")?,
        };
        Ok(GroupSegmentCollector {
            seg_ord,
            mode: self.mode,
            sort: self.sort,
            id_col,
            sort_col,
            key_col,
            shared: Arc::clone(&self.shared),
            buffer: Vec::with_capacity(GROUP_FLUSH_BUFFER),
        })
    }

    fn requires_scoring(&self) -> bool {
        matches!(self.sort, GroupSort::ScoreDesc)
    }

    fn merge_fruits(&self, _per_segment: Vec<()>) -> tantivy::Result<()> {
        Ok(())
    }
}

impl SegmentCollector for GroupSegmentCollector {
    type Fruit = ();

    fn collect(&mut self, doc_id: DocId, score: Score) {
        let id = self.id_col.first(doc_id).unwrap_or_default();
        let sort_val = match self.sort {
            GroupSort::IdAsc => id,
            GroupSort::GenerationAsc => self
                .sort_col
                .as_ref()
                .and_then(|c| c.first(doc_id))
                .unwrap_or(u64::MAX),
            GroupSort::ScoreDesc => u64::from(!score.to_bits()),
        };
        let key = match self.mode {
            ResultGrouping::SameSection => (0u8, self.key_col.first(doc_id).unwrap_or(id)),
            ResultGrouping::IdenticalText => {
                let hash = self.key_col.first(doc_id).unwrap_or(0);
                if hash == 0 {
                    (2u8, id)
                } else {
                    (1u8, hash)
                }
            }
        };
        let entry: GroupEntry = (sort_val, id, DocAddress::new(self.seg_ord, doc_id));
        self.buffer.push((key, entry));
        if self.buffer.len() >= GROUP_FLUSH_BUFFER {
            self.flush();
        }
    }

    fn harvest(mut self) {
        self.flush();
    }
}

// ── BookFingerprintCollector ───────────────────────────────────────────────────

/// Collects a u64 fast-field value (`hash_column`) per `filePath` fast field.
/// Per-segment work uses term ordinals; strings are decoded only in harvest().
/// Documents of the same book that disagree on the hash collapse to 0
/// ("unverifiable"), and 0 wins over any value when merging segments.
struct BookFingerprintCollector {
    hash_column: &'static str,
}

struct BookFingerprintSegmentCollector {
    str_col: Option<tantivy::columnar::StrColumn>,
    hash_col: Option<tantivy::columnar::Column<u64>>,
    fingerprints: HashMap<u64, u64>,
}

impl Collector for BookFingerprintCollector {
    type Fruit = HashMap<String, u64>;
    type Child = BookFingerprintSegmentCollector;

    fn for_segment(
        &self,
        _seg_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> tantivy::Result<BookFingerprintSegmentCollector> {
        let str_col = reader.fast_fields().str("filePath")?;
        // אינדקסים מלפני הוספת השדה: אין עמודה — כל הספרים "לא ניתנים לאימות".
        let hash_col = reader.fast_fields().u64(self.hash_column).ok();
        Ok(BookFingerprintSegmentCollector {
            str_col,
            hash_col,
            fingerprints: HashMap::new(),
        })
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(
        &self,
        per_segment: Vec<tantivy::Result<HashMap<String, u64>>>,
    ) -> tantivy::Result<HashMap<String, u64>> {
        let mut merged: HashMap<String, u64> = HashMap::new();
        for seg_result in per_segment {
            for (path, hash) in seg_result? {
                merged
                    .entry(path)
                    .and_modify(|existing| {
                        if *existing != hash {
                            *existing = 0;
                        }
                    })
                    .or_insert(hash);
            }
        }
        Ok(merged)
    }
}

// ── SingleBookFingerprintCollector ────────────────────────────────────────────

/// The `textHash` of the documents a single-book query matched. Documents that
/// disagree collapse to 0 ("unverifiable"), and 0 wins over any value — same
/// semantics as [`BookFingerprintCollector`], without touching `filePath`:
/// the query already restricts the documents to one book.
struct SingleBookFingerprintCollector;

struct SingleBookFingerprintSegmentCollector {
    hash_col: Option<tantivy::columnar::Column<u64>>,
    fingerprint: Option<u64>,
}

impl Collector for SingleBookFingerprintCollector {
    type Fruit = u64;
    type Child = SingleBookFingerprintSegmentCollector;

    fn for_segment(
        &self,
        _seg_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> tantivy::Result<SingleBookFingerprintSegmentCollector> {
        Ok(SingleBookFingerprintSegmentCollector {
            // אינדקסים מלפני הוספת השדה: אין עמודה — "לא ניתן לאימות".
            hash_col: reader.fast_fields().u64("textHash").ok(),
            fingerprint: None,
        })
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(&self, per_segment: Vec<Option<u64>>) -> tantivy::Result<u64> {
        let mut merged: Option<u64> = None;
        for hash in per_segment.into_iter().flatten() {
            match merged {
                None => merged = Some(hash),
                Some(existing) if existing != hash => return Ok(0),
                Some(_) => {}
            }
        }
        Ok(merged.unwrap_or(0))
    }
}

impl SegmentCollector for SingleBookFingerprintSegmentCollector {
    type Fruit = Option<u64>;

    fn collect(&mut self, doc_id: DocId, _score: Score) {
        let hash = self
            .hash_col
            .as_ref()
            .and_then(|col| col.first(doc_id))
            .unwrap_or(0);
        match self.fingerprint {
            None => self.fingerprint = Some(hash),
            Some(existing) if existing != hash => self.fingerprint = Some(0),
            Some(_) => {}
        }
    }

    fn harvest(self) -> Option<u64> {
        self.fingerprint
    }
}

impl SegmentCollector for BookFingerprintSegmentCollector {
    type Fruit = tantivy::Result<HashMap<String, u64>>;

    fn collect(&mut self, doc_id: DocId, _score: Score) {
        let Some(str_col) = &self.str_col else {
            return;
        };
        let Some(term_ord) = str_col.term_ords(doc_id).next() else {
            return;
        };
        let hash = self
            .hash_col
            .as_ref()
            .and_then(|col| col.first(doc_id))
            .unwrap_or(0);
        self.fingerprints
            .entry(term_ord)
            .and_modify(|existing| {
                if *existing != hash {
                    *existing = 0;
                }
            })
            .or_insert(hash);
    }

    fn harvest(self) -> tantivy::Result<HashMap<String, u64>> {
        let Some(col) = self.str_col else {
            return Ok(HashMap::new());
        };
        let mut result = HashMap::with_capacity(self.fingerprints.len());
        let mut buf = String::new();
        for (term_ord, hash) in self.fingerprints {
            buf.clear();
            if col.ord_to_str(term_ord, &mut buf)? {
                result.insert(buf.clone(), hash);
            }
        }
        Ok(result)
    }
}

/// Segments `optimize` merges into one: every delete-heavy segment, plus the
/// smallest others until the count fits the cap. `None` when nothing qualifies.
fn select_segments_to_compact(metas: &[SegmentMeta]) -> Option<Vec<SegmentId>> {
    let mut by_size: Vec<&SegmentMeta> = metas.iter().collect();
    by_size.sort_by_key(|m| m.num_docs());

    let mut chosen: Vec<SegmentId> = by_size
        .iter()
        .filter(|m| {
            m.max_doc() > 0
                && f64::from(m.num_deleted_docs()) / f64::from(m.max_doc())
                    > OPTIMIZE_COMPACT_DELETE_RATIO
        })
        .map(|m| m.id())
        .collect();

    // Merging n segments leaves len - n + 1; the smallest go first so the big
    // ones are naturally excluded.
    let count_after = |chosen: usize| {
        if chosen == 0 {
            metas.len()
        } else {
            metas.len() - chosen + 1
        }
    };
    for meta in &by_size {
        if count_after(chosen.len()) <= MAX_SEGMENTS_AFTER_OPTIMIZE {
            break;
        }
        if !chosen.contains(&meta.id()) {
            chosen.push(meta.id());
        }
    }
    (!chosen.is_empty()).then_some(chosen)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn make_engine() -> (SearchEngine, TempDir) {
        let dir = TempDir::new().unwrap();
        let engine = SearchEngine::new(dir.path().to_str().unwrap());
        (engine, dir)
    }

    fn dir_path_string(dir: &TempDir) -> String {
        dir.path().to_str().unwrap().to_string()
    }

    #[allow(clippy::too_many_arguments)]
    fn search_advanced_default(
        engine: &SearchEngine,
        query: String,
        facets: Vec<String>,
        limit: u32,
        offset: u32,
        distance: u32,
        custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        order: ResultsOrder,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
    ) -> Result<Vec<SearchResult>> {
        let negative_scope = same_search_scope(&scope);
        engine.search_advanced(
            query,
            String::new(),
            facets,
            limit,
            offset,
            distance,
            distance,
            custom_spacing,
            HashMap::new(),
            alternative_words,
            HashMap::new(),
            search_options,
            HashMap::new(),
            order,
            match_nikud,
            match_taamim,
            scope,
            negative_scope,
            None,
            None,
            None,
        )
    }

    fn same_search_scope(scope: &SearchScope) -> SearchScope {
        match scope {
            SearchScope::WordDistance => SearchScope::WordDistance,
            SearchScope::SameParagraph => SearchScope::SameParagraph,
            SearchScope::SameSection => SearchScope::SameSection,
        }
    }

    fn count_advanced_default(
        engine: &SearchEngine,
        query: String,
        facets: Vec<String>,
        distance: u32,
        custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
    ) -> Result<u32> {
        let negative_scope = same_search_scope(&scope);
        engine.count_advanced(
            query,
            String::new(),
            facets,
            distance,
            distance,
            custom_spacing,
            HashMap::new(),
            alternative_words,
            HashMap::new(),
            search_options,
            HashMap::new(),
            match_nikud,
            match_taamim,
            scope,
            negative_scope,
            None,
            None,
        )
    }

    fn count_by_book_advanced_default(
        engine: &SearchEngine,
        query: String,
        facets: Vec<String>,
        distance: u32,
        custom_spacing: HashMap<String, String>,
        alternative_words: HashMap<u32, Vec<String>>,
        search_options: HashMap<String, HashMap<String, bool>>,
        match_nikud: bool,
        match_taamim: bool,
        scope: SearchScope,
    ) -> Result<HashMap<String, u32>> {
        let negative_scope = same_search_scope(&scope);
        engine.count_by_book_advanced(
            query,
            String::new(),
            facets,
            distance,
            distance,
            custom_spacing,
            HashMap::new(),
            alternative_words,
            HashMap::new(),
            search_options,
            HashMap::new(),
            match_nikud,
            match_taamim,
            scope,
            negative_scope,
            None,
            None,
        )
    }

    /// Writes a tiny `lexical.db` into `dir`: lemma "הלכ" with surfaces
    /// "הלכתי"/"הולכ" (folded, as the real DB stores them) and returns its path.
    fn make_lexical_db(dir: &TempDir) -> String {
        let path = dir.path().join("lexical.db");
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE base (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT NOT NULL UNIQUE);
            CREATE TABLE surface (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT NOT NULL UNIQUE, base_id INTEGER NOT NULL REFERENCES base(id), notes TEXT);
            CREATE TABLE variant (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT NOT NULL UNIQUE);
            CREATE TABLE surface_variant (surface_id INTEGER NOT NULL REFERENCES surface(id), variant_id INTEGER NOT NULL REFERENCES variant(id), PRIMARY KEY (surface_id, variant_id));
            INSERT INTO base (id, value) VALUES (1, 'הלכ'), (2, 'ישנ'), (3, 'אדמור');
            INSERT INTO surface (id, value, base_id) VALUES
                (1, 'הלכתי', 1),
                (2, 'הולכ', 1),
                (3, 'לכו', 1),
                (4, 'הולכימ', 1),
                (5, 'לישונ', 2),
                (6, 'ישנ', 2),
                (7, 'בלשונ', 2),
                -- מסמן את הפער השיורי של אופציה A (ראו §5.2 בתכנון): ה-DB
                -- מחזיק צורות נקיות בלבד, בעוד שהאינדקס עשוי לשאת אדמו"ר.
                (8, 'אדמור', 3),
                (9, 'אדמורימ', 3);
            "#,
        )
        .unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn new_writes_index_metadata_sidecar() {
        let (_engine, dir) = make_engine();
        let metadata_path = index_metadata_path(dir.path());
        assert!(metadata_path.exists());

        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(compatibility.compatible);
        assert_eq!(compatibility.status, "compatible");
        assert_eq!(
            compatibility.found_schema_version,
            Some(INDEX_SCHEMA_VERSION)
        );
    }

    #[test]
    fn sidecar_version_match_but_schema_drift_requires_rebuild() {
        // שחזור התקלה מהשטח: אינדקס שנבנה בגרסת-ביניים של אותה
        // schema_version (למשל `text` עם fast=true לפני ההסרה) — הקובץ
        // הצדדי מצהיר על הגרסה הנכונה, אבל open_or_create היה נופל על
        // SchemaError. הבדיקה חייבת לדרוש בנייה מחדש, לא "תואם".
        let dir = TempDir::new().unwrap();
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field(
            "text",
            TextOptions::default()
                .set_indexing_options(
                    TextFieldIndexing::default()
                        .set_tokenizer("hebrew")
                        .set_index_option(IndexRecordOption::WithFreqsAndPositions),
                )
                .set_stored()
                .set_fast(None),
        );
        let drifted = schema_builder.build();
        Index::create_in_dir(dir.path(), drifted).unwrap();
        // הקובץ הצדדי נכתב במפורש עם הגרסה הנוכחית — כמו אינדקס שנבנה
        // ע"י גרסת-הביניים עצמה, שחתמה "3" עם הסכימה הישנה.
        write_current_index_metadata(dir.path()).unwrap();

        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(!compatibility.compatible);
        assert_eq!(compatibility.status, "rebuild_required");
        assert_eq!(
            compatibility.found_schema_version,
            Some(INDEX_SCHEMA_VERSION)
        );
        assert!(compatibility
            .reason
            .unwrap()
            .contains("differs from the engine schema"));
    }

    #[test]
    fn valid_sidecar_without_tantivy_meta_requires_rebuild() {
        // sidecar תקין לבדו לא מספיק: בלי meta.json של tantivy (חסר או
        // פגום) פתיחת האינדקס תיכשל, ולכן הבדיקה חייבת לדרוש בנייה מחדש
        // ולא להחזיר "תואם" רק על סמך ההצהרה בקובץ הצדדי.
        let dir = TempDir::new().unwrap();
        write_current_index_metadata(dir.path()).unwrap();

        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(!compatibility.compatible);
        assert_eq!(compatibility.status, "rebuild_required");
        assert_eq!(
            compatibility.found_schema_version,
            Some(INDEX_SCHEMA_VERSION)
        );
        assert!(compatibility
            .reason
            .unwrap()
            .contains("missing or unreadable"));

        // אותו דין ל-meta.json קיים אך פגום.
        fs::write(dir.path().join("meta.json"), "not json").unwrap();
        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(!compatibility.compatible);
        assert_eq!(compatibility.status, "rebuild_required");
        assert!(compatibility.reason.unwrap().contains("not valid JSON"));
    }

    #[test]
    fn missing_sidecar_uses_tantivy_schema_fallback() {
        let (_engine, dir) = make_engine();
        fs::remove_file(index_metadata_path(dir.path())).unwrap();

        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(compatibility.compatible);
        assert_eq!(compatibility.status, "legacy_compatible");
        assert_eq!(
            compatibility.found_schema_version,
            Some(INDEX_SCHEMA_VERSION)
        );
    }

    #[test]
    fn old_sidecar_schema_requires_rebuild() {
        let dir = TempDir::new().unwrap();
        let mut metadata = current_index_metadata();
        metadata.schema_version = INDEX_SCHEMA_VERSION - 1;
        fs::write(
            index_metadata_path(dir.path()),
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .unwrap();

        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(!compatibility.compatible);
        assert_eq!(compatibility.status, "rebuild_required");
        assert_eq!(
            compatibility.found_schema_version,
            Some(INDEX_SCHEMA_VERSION - 1)
        );
    }

    #[test]
    fn future_sidecar_schema_marks_engine_too_old() {
        let dir = TempDir::new().unwrap();
        let mut metadata = current_index_metadata();
        metadata.schema_version = INDEX_SCHEMA_VERSION + 1;
        fs::write(
            index_metadata_path(dir.path()),
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .unwrap();

        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(!compatibility.compatible);
        assert_eq!(compatibility.status, "engine_too_old");
        assert_eq!(
            compatibility.found_schema_version,
            Some(INDEX_SCHEMA_VERSION + 1)
        );
    }

    #[test]
    fn legacy_tantivy_schema_without_indexed_id_requires_rebuild() {
        let dir = TempDir::new().unwrap();
        let tantivy_metadata = json!({
            "schema": [
                {
                    "name": "id",
                    "type": "u64",
                    "options": {
                        "indexed": false,
                        "fast": true,
                        "stored": true
                    }
                }
            ]
        });
        fs::write(dir.path().join("meta.json"), tantivy_metadata.to_string()).unwrap();

        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(!compatibility.compatible);
        assert_eq!(compatibility.status, "rebuild_required");
        assert_eq!(compatibility.found_schema_version, Some(1));
    }

    #[test]
    fn legacy_schema_with_current_id_but_old_file_path_requires_rebuild() {
        let dir = TempDir::new().unwrap();
        {
            // `id` matches the current shape, but `filePath` lacks FAST (and
            // is tokenized) — the engine could not open this index, so the
            // full-schema check must fail it instead of passing on `id` alone.
            let mut b = Schema::builder();
            b.add_text_field("text", TEXT | STORED | FAST);
            b.add_text_field("reference", STORED);
            b.add_text_field(
                "title",
                TextOptions::default()
                    .set_indexing_options(
                        TextFieldIndexing::default()
                            .set_tokenizer("raw")
                            .set_fieldnorms(false),
                    )
                    .set_stored(),
            );
            b.add_u64_field("id", STORED | FAST | INDEXED);
            b.add_u64_field("segment", STORED);
            b.add_bool_field("isPdf", STORED);
            b.add_text_field("filePath", TEXT | STORED);
            b.add_facet_field("topics", FacetOptions::default());
            let old_schema = b.build();
            let mmap = MmapDirectory::open(dir.path()).unwrap();
            Index::open_or_create(mmap, old_schema).unwrap();
        }

        let compatibility = check_index_compatibility(dir_path_string(&dir));
        assert!(!compatibility.compatible);
        assert_eq!(compatibility.status, "rebuild_required");
    }

    fn add(engine: &mut SearchEngine, id: u64, text: &str, file_path: &str) {
        engine
            .add_document(
                id, "title", "ref", "/root", text, 0, false, file_path, None, None, None,
            )
            .unwrap();
    }

    #[test]
    fn semantic_status_is_explicit_before_configuration() {
        let (engine, _dir) = make_engine();
        let status = engine.semantic_status();
        assert!(!status.enabled);
        assert!(!status.available);
        assert!(status.last_error.is_some());
    }

    #[test]
    fn hybrid_request_falls_back_to_ranked_lexical_results_with_reason() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 41, "שלום עולם", "/books/a.txt");
        engine.commit().unwrap();

        let response = engine
            .search_semantic(
                "שלום".to_string(),
                Vec::new(),
                10,
                0,
                SemanticLexicalMode::Exact,
                1,
                SemanticRetrievalMode::Hybrid,
                None,
                false,
                false,
            )
            .unwrap();

        assert_eq!(response.executed_mode, SemanticExecutedMode::LexicalOnly);
        assert_eq!(response.lexical_total_count, 1);
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].id, 41);
        assert_eq!(response.results[0].source, SemanticResultSource::Lexical);
        assert!(response.fallback_reason.is_some());
    }

    fn disable_auto_merge(engine: &SearchEngine) {
        engine
            .index_writer
            .as_ref()
            .unwrap()
            .set_merge_policy(Box::new(NoMergePolicy));
    }

    fn search_ids(engine: &mut SearchEngine, term: &str) -> Vec<u64> {
        engine
            .search(
                vec![term.to_string()],
                vec!["/root".to_string()],
                100,
                0,
                0,
                100,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|result| result.id)
            .collect()
    }

    #[test]
    fn generation_order_can_prioritize_sources_before_commentaries() {
        let (mut engine, _dir) = make_engine();
        engine
            .add_document(
                10,
                "פירוש מוקדם בקטלוג",
                "ref",
                "/root",
                "שלום",
                0,
                false,
                "/books/commentary.txt",
                None,
                Some(65),
                None,
            )
            .unwrap();
        engine
            .add_document(
                30,
                "מקור שני",
                "ref",
                "/root",
                "שלום",
                0,
                false,
                "/books/source-b.txt",
                None,
                Some(2),
                None,
            )
            .unwrap();
        engine
            .add_document(
                20,
                "מקור ראשון",
                "ref",
                "/root",
                "שלום",
                0,
                false,
                "/books/source-a.txt",
                None,
                Some(2),
                None,
            )
            .unwrap();
        engine.commit().unwrap();

        let by_generation: Vec<u64> = engine
            .search_exact(
                "שלום".to_string(),
                vec!["/root".to_string()],
                10,
                0,
                ResultsOrder::Generation,
                false,
                false,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|result| result.id)
            .collect();

        assert_eq!(by_generation, vec![20, 30, 10]);
    }

    #[test]
    fn add_text_book_builds_reference_trail_ids_and_fingerprint() {
        let (mut engine, _dir) = make_engine();
        let text =
            "<h1>ספר בראשית</h1>\n<h2>פרק א</h2>\nבְּרֵאשִׁית ברא אלהים\n<h2>פרק ב</h2>\nויכלו השמים";
        let added = engine
            .add_text_book(
                "בראשית".to_string(),
                "/root".to_string(),
                "/books/bereshit.txt".to_string(),
                5,
                DEFAULT_GENERATION_ORDER,
                text.to_string(),
                None,
            )
            .unwrap();
        assert_eq!(added, 5);
        engine.commit().unwrap();

        let results = engine
            .search_exact(
                "ויכלו".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        let hit = &results[0];
        // הכותרת החדשה של פרק ב החליפה את פרק א ב-trail (אותו prefix "<h2>").
        assert_eq!(hit.reference, "ספר בראשית, פרק ב");
        assert_eq!(hit.segment, 4);
        // id = ((catalogue_order+1) << 32) + ordinal+1, כמו ב-Dart.
        assert_eq!(hit.id, ((5u64 + 1) << 32) + 5);
        assert_eq!(hit.title, "בראשית");
        assert!(!hit.is_pdf);

        // הטקסט המאונדקס מנורמל (הניקוד הוסר) והחיפוש מוצא אותו.
        let nikud_hit = engine
            .search_exact(
                "בראשית ברא".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(nikud_hit.len(), 1);

        // טביעת האצבע נחתמה על הספר, זהה לחישוב הציבורי הקנוני
        // (טקסט + metadata).
        let fingerprints = engine.get_book_fingerprints().unwrap();
        assert_eq!(
            fingerprints.get("/books/bereshit.txt"),
            Some(&compute_book_fingerprint(
                text.to_string(),
                "בראשית".to_string(),
                "/root".to_string(),
                5,
                DEFAULT_GENERATION_ORDER,
                None,
            ))
        );
        // שינוי metadata בלבד (סדר דורות) משנה את החתימה — הספר יזוהה
        // כדורש אינדוקס-מחדש.
        assert_ne!(
            compute_book_fingerprint(
                text.to_string(),
                "בראשית".to_string(),
                "/root".to_string(),
                5,
                DEFAULT_GENERATION_ORDER,
                None,
            ),
            compute_book_fingerprint(
                text.to_string(),
                "בראשית".to_string(),
                "/root".to_string(),
                5,
                7,
                None,
            ),
        );
        // סדר ה-extra_facets אינו משנה את החתימה.
        assert_eq!(
            compute_book_fingerprint(
                text.to_string(),
                "בראשית".to_string(),
                "/root".to_string(),
                5,
                DEFAULT_GENERATION_ORDER,
                Some(vec![
                    "/author/רש\"י".to_string(),
                    "/era/ראשונים".to_string()
                ]),
            ),
            compute_book_fingerprint(
                text.to_string(),
                "בראשית".to_string(),
                "/root".to_string(),
                5,
                DEFAULT_GENERATION_ORDER,
                Some(vec![
                    "/era/ראשונים".to_string(),
                    "/author/רש\"י".to_string()
                ]),
            ),
        );

        // חתימת הטקסט-בלבד נחתמה גם היא, זהה לחישוב הציבורי — ובניגוד
        // לקנונית, אינה תלויה בסדר הקטלוגי או בשאר ה-metadata.
        let text_fingerprints = engine.get_book_text_fingerprints().unwrap();
        assert_eq!(
            text_fingerprints.get("/books/bereshit.txt"),
            Some(&compute_content_fingerprint(text.to_string()))
        );
    }

    #[test]
    fn breaking_tag_in_a_line_keeps_the_words_separate() {
        // Otzaria/otzaria#949: `<br>` בתוך שורה (ספרים אישיים מומרים) הדביק
        // את שתי המילים לטוקן אחד — הן הוצגו דבוקות בתוצאות, וחיפוש של כל
        // אחת מהן החטיא.
        let (mut engine, _dir) = make_engine();
        engine
            .add_text_book(
                "חוזק יד".to_string(),
                "/root".to_string(),
                "/books/chozek.txt".to_string(),
                1,
                DEFAULT_GENERATION_ORDER,
                "יותר בכבוד המורים<br>כי אב הביאו לעולם".to_string(),
                None,
            )
            .unwrap();
        engine.commit().unwrap();

        for query in ["המורים", "כי", "המורים כי"] {
            let results = engine
                .search_exact(
                    query.to_string(),
                    vec![],
                    10,
                    0,
                    ResultsOrder::Catalogue,
                    false,
                    false,
                    None,
                )
                .unwrap();
            assert_eq!(results.len(), 1, "query {query:?}");
            // הטקסט השמור — שממנו נבנה קטע התוצאה — מופרד ברווח (תגי
            // ההדגשה של המנוע עשויים לעטוף את ההתאמות).
            let plain = results[0]
                .text
                .replace("<font color=red>", "")
                .replace("</font>", "");
            assert!(plain.contains("המורים כי"), "stored text: {plain}");
        }
    }

    #[test]
    fn add_document_and_batch_normalize_like_add_text_book() {
        // ה-API הישירים חשופים ב-FFI — ההנחה "הקלט כבר מנורמל" נאכפת:
        // HTML מוסר, ניקוד מוסר מהשדה הרגיל, והעותק המנוקד נבנה מהגולמי.
        let raw = "<b>בְּרֵאשִׁית</b>  בָּרָא אלהים";
        let (mut engine, _dir) = make_engine();
        engine
            .add_document(
                1,
                "title",
                "ref",
                "/root",
                raw,
                0,
                false,
                "/books/a.txt",
                None,
                None,
                None,
            )
            .unwrap();
        engine
            .add_documents_batch(vec![DocumentInput {
                id: 2,
                title: "title".to_string(),
                reference: "ref".to_string(),
                topics: "/root".to_string(),
                text: raw.to_string(),
                segment: 1,
                is_pdf: false,
                file_path: "/books/b.txt".to_string(),
                content_hash: None,
                text_hash: None,
                text_vocalized: Some("<b>בְּרֵאשִׁית</b>  בָּרָא אלהים".to_string()),
                section_id: None,
                generation_order: None,
                extra_facets: None,
            }])
            .unwrap();
        engine
            .upsert_documents_batch(vec![DocumentInput {
                id: 3,
                title: "title".to_string(),
                reference: "ref".to_string(),
                topics: "/root".to_string(),
                text: raw.to_string(),
                segment: 2,
                is_pdf: false,
                file_path: "/books/c.txt".to_string(),
                content_hash: None,
                text_hash: None,
                text_vocalized: Some("<b>בְּרֵאשִׁית</b>  בָּרָא אלהים".to_string()),
                section_id: None,
                generation_order: None,
                extra_facets: None,
            }])
            .unwrap();
        engine.commit().unwrap();

        // שלושת המסלולים מאונדקסים ומאוחסנים בצורה המנורמלת.
        let results = engine
            .search_exact(
                "בראשית ברא".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 3);
        for hit in &results {
            // הטקסט השמור מנורמל (בלי ה-HTML המקורי ובלי ניקוד בשדה
            // הרגיל); תגי ה-font הם עיטור ההדגשה של תוצאת החיפוש.
            let clean = hit
                .text
                .replace("<font color=red>", "")
                .replace("</font>", "");
            assert_eq!(clean, "בראשית ברא אלהים");
        }

        // העותקים המנוקדים נורמלו גם הם (ה-HTML הוסר): מסלולי ה-batch
        // (add/upsert) דרך הנרמול הנאכף על text_vocalized שסופק, ומסלול
        // add_document דרך העותק שהוא בונה בעצמו מהקלט הגולמי — חיפוש
        // מנוקד מוצא את כולן.
        let vocalized = engine
            .search_exact(
                "בְּרֵאשִׁית".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                true,
                false,
                None,
            )
            .unwrap();
        assert_eq!(vocalized.len(), 3);
    }

    #[test]
    fn analyzers_drop_tokens_longer_than_cap() {
        // RemoveLongFilter על כל האנליזטורים: זבל base64 (נמדדו ריצות של
        // אלפי תווים בקורפוס) לא נכנס למילון הטרמים — וגם צד השאילתה מפיל
        // את הטוקן, כך שספירת PhraseQuery נשארת עקבית עם האינדקס.
        let (mut engine, _dir) = make_engine();
        let garbage = "א".repeat(80); // 160 בייט — מעל התקרה
        let long_but_legit = "א".repeat(60); // 120 בייט — מתחת לתקרה
        add(
            &mut engine,
            1,
            &format!("שלום {garbage} עולם"),
            "/books/junk.txt",
        );
        engine.commit().unwrap();

        // הטוקן הארוך לא נטמע; המילים סביבו כן.
        assert_eq!(
            engine
                .index_token_texts(&format!("שלום {garbage}"))
                .unwrap(),
            vec!["שלום"]
        );
        assert_eq!(engine.index_token_texts(&long_but_legit).unwrap().len(), 1);
        assert_eq!(search_ids(&mut engine, "שלום"), vec![1]);
        assert_eq!(search_ids(&mut engine, &garbage), Vec::<u64>::new());
    }

    #[test]
    fn add_text_book_bytes_matches_string_path() {
        // מסלול הבייטים (SQLite BLOB → Uint8List) חייב לייצר בדיוק את אותם
        // מסמכים ואותה טביעת אצבע כמו מסלול ה-String.
        let text =
            "<h1>ספר בראשית</h1>\n<h2>פרק א</h2>\nבְּרֵאשִׁית ברא אלהים\n<h2>פרק ב</h2>\nויכלו השמים";
        let (mut engine, _dir) = make_engine();
        let added = engine
            .add_text_book_bytes(
                "בראשית".to_string(),
                "/root".to_string(),
                "/books/bereshit.txt".to_string(),
                5,
                DEFAULT_GENERATION_ORDER,
                text.as_bytes().to_vec(),
                None,
            )
            .unwrap();
        assert_eq!(added, 5);
        engine.commit().unwrap();

        let results = engine
            .search_exact(
                "ויכלו".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].reference, "ספר בראשית, פרק ב");
        assert_eq!(results[0].id, ((5u64 + 1) << 32) + 5);

        let fingerprints = engine.get_book_fingerprints().unwrap();
        assert_eq!(
            fingerprints.get("/books/bereshit.txt"),
            Some(&compute_book_fingerprint(
                text.to_string(),
                "בראשית".to_string(),
                "/root".to_string(),
                5,
                DEFAULT_GENERATION_ORDER,
                None,
            ))
        );
        let text_fingerprints = engine.get_book_text_fingerprints().unwrap();
        assert_eq!(
            text_fingerprints.get("/books/bereshit.txt"),
            Some(&compute_content_fingerprint(text.to_string()))
        );
    }

    #[test]
    fn add_text_book_empty_text_adds_nothing() {
        let (mut engine, _dir) = make_engine();
        let added = engine
            .add_text_book(
                "ריק".to_string(),
                "/root".to_string(),
                "/books/empty.txt".to_string(),
                1,
                DEFAULT_GENERATION_ORDER,
                String::new(),
                None,
            )
            .unwrap();
        assert_eq!(added, 0);
        engine.commit().unwrap();
        assert_eq!(engine.get_document_count(), 0);
    }

    #[test]
    fn add_pdf_book_filters_garbage_and_encodes_ids() {
        let (mut engine, _dir) = make_engine();
        let pages = vec![
            PdfPageInput {
                reference: "ספר, עמוד 1".to_string(),
                text: "שורה ראשונה בעמוד\n\n≡≡≡ ∴∴∴ ⊕⊗⊘".to_string(),
                page_index: 0,
            },
            PdfPageInput {
                reference: "ספר, עמוד 2".to_string(),
                text: "בְּרֵאשִׁית ברא אלהים".to_string(),
                page_index: 1,
            },
        ];
        let added = engine
            .add_pdf_book(
                "ספר".to_string(),
                "/root".to_string(),
                "C:/books/sefer.pdf".to_string(),
                5,
                DEFAULT_GENERATION_ORDER,
                pages,
                None,
            )
            .unwrap();
        // השורה הריקה ושורת הסימנים סוננו כזבל — נותרו שתי שורות תוכן.
        assert_eq!(added, 2);
        engine.commit().unwrap();

        // הטקסט מנורמל לפני האינדוקס — שאילתה ללא ניקוד מוצאת אותו.
        let results = engine
            .search_exact(
                "בראשית ברא".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        let hit = &results[0];
        assert_eq!(hit.reference, "ספר, עמוד 2");
        // segment = אינדקס העמוד; ordinal רץ על שורות התוכן בלבד.
        assert_eq!(hit.segment, 1);
        assert_eq!(hit.id, ((5u64 + 1) << 32) + 2);
        assert!(hit.is_pdf);

        // ל-PDF אין טביעת אצבע — contentHash נחתם כ-0 (לא נרשם).
        let fingerprints = engine.get_book_fingerprints().unwrap();
        assert_eq!(fingerprints.get("C:/books/sefer.pdf"), Some(&0u64));
    }

    #[test]
    fn bulk_indexing_skips_merges_and_optimize_bounds_segments() {
        let (mut engine, _dir) = make_engine();
        engine.set_bulk_indexing(true).unwrap();
        // כמה commit-ים ⇒ כמה סגמנטים; ב-bulk אין מיזוג רקע שמאחד אותם.
        for i in 0..3u64 {
            add(&mut engine, i + 1, "שלום עולם", "/books/a.txt");
            engine.commit().unwrap();
        }
        assert!(engine.index.searchable_segment_ids().unwrap().len() > 1);

        engine.set_bulk_indexing(false).unwrap();
        engine.optimize().unwrap();
        assert!(
            engine.index.searchable_segment_ids().unwrap().len() <= MAX_SEGMENTS_AFTER_OPTIMIZE
        );
        assert_eq!(engine.get_document_count(), 3);
    }

    /// כל commit מפזר מסמכים על כמה סגמנטים (thread לכל סגמנט); הבדיקות
    /// שצריכות "סגמנט גדול אחד" מאחדות אותו כאן, מחוץ ל-optimize הנבדק.
    fn collapse_to_one_segment(engine: &mut SearchEngine) {
        let live = engine.take_writer().unwrap();
        drop(live);
        let mut writer = engine.open_writer_no_merge().unwrap();
        let ids = engine.index.searchable_segment_ids().unwrap();
        writer.merge(&ids).wait().unwrap();
        writer.wait_merging_threads().unwrap();
        engine.restore_writer().unwrap();
        engine.index_reader.reload().unwrap();
        assert_eq!(engine.index.searchable_segment_ids().unwrap().len(), 1);
    }

    fn count_hits(engine: &SearchEngine, word: &str) -> usize {
        engine.count(vec![word.to_string()], &[], 0, 1).unwrap() as usize
    }

    #[test]
    fn optimize_merges_smallest_segments_down_to_cap() {
        let (mut engine, _dir) = make_engine();
        // bulk נשאר פעיל גם ב-optimize — אחרת LogMergePolicy היה מאחד ברקע
        // והבדיקה לא הייתה מודדת את optimize עצמו.
        engine.set_bulk_indexing(true).unwrap();
        let total = MAX_SEGMENTS_AFTER_OPTIMIZE as u64 + 4;
        for i in 0..total {
            add(&mut engine, i + 1, "שלום עולם", "/books/a.txt");
            engine.commit().unwrap();
        }
        assert_eq!(
            engine.index.searchable_segment_ids().unwrap().len(),
            total as usize
        );

        engine.optimize().unwrap();
        assert_eq!(
            engine.index.searchable_segment_ids().unwrap().len(),
            MAX_SEGMENTS_AFTER_OPTIMIZE
        );
        assert_eq!(engine.get_document_count(), total);
        assert_eq!(count_hits(&engine, "שלום"), total as usize);
    }

    fn count_idx_files(dir: &TempDir) -> usize {
        std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension() == Some("idx".as_ref()))
            .count()
    }

    fn dir_size_bytes(dir: &TempDir) -> u64 {
        std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().metadata().unwrap().len())
            .sum()
    }

    #[test]
    fn optimize_garbage_collects_merged_away_segment_files() {
        let (mut engine, dir) = make_engine();
        engine.set_bulk_indexing(true).unwrap();
        let total = MAX_SEGMENTS_AFTER_OPTIMIZE as u64 + 4;
        for i in 0..total {
            add(&mut engine, i + 1, "שלום עולם", "/books/a.txt");
            engine.commit().unwrap();
        }
        // Pin the pre-optimize segments the way a live search would.
        engine.index_reader.reload().unwrap();
        assert_eq!(count_idx_files(&dir), total as usize);
        let size_before = dir_size_bytes(&dir);

        engine.optimize().unwrap();
        let searchable = engine.index.searchable_segment_ids().unwrap().len();
        assert_eq!(searchable, MAX_SEGMENTS_AFTER_OPTIMIZE);
        assert_eq!(
            count_idx_files(&dir),
            searchable,
            "merged-away segment files must be garbage-collected"
        );
        assert!(
            dir_size_bytes(&dir) <= size_before,
            "index directory must not grow after optimize"
        );
    }

    #[test]
    fn optimize_leaves_large_segment_untouched_next_to_tiny_one() {
        let (mut engine, _dir) = make_engine();
        engine.set_bulk_indexing(true).unwrap();
        for i in 0..500u64 {
            add(&mut engine, i + 1, "בראשית ברא", "/books/big.txt");
        }
        engine.commit().unwrap();
        collapse_to_one_segment(&mut engine);
        let big_id = engine.index.searchable_segment_ids().unwrap()[0];

        add(&mut engine, 1_000, "שלום עולם", "/books/small.txt");
        engine.commit().unwrap();
        assert_eq!(engine.index.searchable_segment_ids().unwrap().len(), 2);

        engine.optimize().unwrap();
        let ids = engine.index.searchable_segment_ids().unwrap();
        assert_eq!(ids.len(), 2);
        // הסגמנט הגדול לא נכתב מחדש — אותו SegmentId עדיין שם.
        assert!(ids.contains(&big_id));
        assert_eq!(engine.get_document_count(), 501);
    }

    #[test]
    fn optimize_compacts_delete_heavy_segment() {
        let (mut engine, _dir) = make_engine();
        engine.set_bulk_indexing(true).unwrap();
        for i in 0..20u64 {
            add(&mut engine, i + 1, "בראשית ברא", "/books/gone.txt");
        }
        add(&mut engine, 100, "שלום עולם", "/books/kept.txt");
        engine.commit().unwrap();
        collapse_to_one_segment(&mut engine);
        let heavy_id = engine.index.searchable_segment_ids().unwrap()[0];

        engine
            .delete_documents_by_file_path("/books/gone.txt")
            .unwrap();
        engine.commit().unwrap();
        let metas = engine.index.searchable_segment_metas().unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].num_deleted_docs(), 20);

        engine.optimize().unwrap();
        let metas = engine.index.searchable_segment_metas().unwrap();
        assert_eq!(metas.len(), 1);
        assert_ne!(metas[0].id(), heavy_id);
        assert_eq!(metas[0].num_deleted_docs(), 0);
        assert_eq!(metas[0].num_docs(), 1);
        assert_eq!(count_hits(&engine, "שלום"), 1);
    }

    #[test]
    fn economy_indexing_swaps_writer_without_losing_pending_docs() {
        let (mut engine, _dir) = make_engine();
        // מסמך שטרם עבר commit חייב לשרוד את החלפת ה-writer.
        add(&mut engine, 1, "בראשית ברא", "/books/a.txt");
        engine.set_economy_indexing(true).unwrap();
        assert_eq!(engine.writer_heap_size, ECONOMY_WRITER_HEAP_SIZE);
        assert_eq!(engine.get_document_count(), 1);

        // גם בדרך חזרה, וגם כשמצב bulk פעיל — ה-writer החדש שומר NoMergePolicy.
        engine.set_bulk_indexing(true).unwrap();
        add(&mut engine, 2, "שלום עולם", "/books/a.txt");
        engine.set_economy_indexing(false).unwrap();
        assert_eq!(engine.writer_heap_size, DEFAULT_WRITER_HEAP_SIZE);
        engine.set_economy_indexing(false).unwrap();
        assert_eq!(engine.get_document_count(), 2);
    }

    #[test]
    fn add_pdf_book_all_garbage_adds_nothing() {
        let (mut engine, _dir) = make_engine();
        let added = engine
            .add_pdf_book(
                "סרוק".to_string(),
                "/root".to_string(),
                "C:/books/scanned.pdf".to_string(),
                1,
                DEFAULT_GENERATION_ORDER,
                vec![PdfPageInput {
                    reference: "סרוק, עמוד 1".to_string(),
                    text: "\n≡≡≡≡≡\n∴ ⊕ ⊗ ⊘ ∴".to_string(),
                    page_index: 0,
                }],
                None,
            )
            .unwrap();
        assert_eq!(added, 0);
        engine.commit().unwrap();
        assert_eq!(engine.get_document_count(), 0);
    }

    #[test]
    fn exact_search_raw_maqaf_query_becomes_phrase() {
        // רגרסיה: strip_nikud לפני הטוקניזציה מחק את המקף והדביק
        // "אשר־שמע" לטרם בודד ("אשרשמע") שאינו קיים באינדקס.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ברוך אֲשֶׁר־שָׁמַע את הדבר", "/books/a.txt");
        add(&mut engine, 2, "אשר לא שמע דבר", "/books/b.txt");
        engine.commit().unwrap();

        let ids: Vec<u64> = engine
            .search_exact(
                "אשר־שמע".to_string(),
                vec![],
                100,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        // phrase של שתי מילים סמוכות — תופס את 1, לא את 2 (מילים מרוחקות).
        assert_eq!(ids, vec![1]);
    }

    // ── Single-word alternation splitting (per-branch DFA) ───────────────

    /// The exact pattern the generator builds for `משה` with "חלק ממילה" +
    /// "שגיאות כתיב": 48 wildcard-wrapped branches, 806 chars. As one regex it
    /// exceeded the upstream tantivy-fst 1 000-state DFA cap (the vendored
    /// copy raises it to 8 192); each branch alone is tiny.
    const BOTH_OPTIONS_PATTERN: &str = "(.{0,3}משה.{0,3}|.{0,3}מסה.{0,3}|.{0,2}משׁה.{0,2}|.{0,2}משׂה.{0,2}|.{0,3}משא.{0,3}|.{0,3}משע.{0,3}|.{0,3}משח.{0,3}|.{0,3}שמה.{0,3}|.{0,3}מהש.{0,3}|.{0,3}שה.{0,3}|.{0,3}מה.{0,3}|.{0,3}מש.{0,3}|.{0,2}ומשה.{0,2}|.{0,2}ימשה.{0,2}|.{0,2}אמשה.{0,2}|.{0,2}המשה.{0,2}|.{0,2}פמשה.{0,2}|.{0,2}למשה.{0,2}|.{0,2}ממשה.{0,2}|.{0,2}נמשה.{0,2}|.{0,2}במשה.{0,2}|.{0,2}כמשה.{0,2}|.{0,2}שמשה.{0,2}|.{0,2}תמשה.{0,2}|.{0,2}רמשה.{0,2}|.{0,2}משהו.{0,2}|.{0,2}משהי.{0,2}|.{0,2}משהא.{0,2}|.{0,2}משהה.{0,2}|.{0,2}משהפ.{0,2}|.{0,2}משהל.{0,2}|.{0,2}משהמ.{0,2}|.{0,2}משהנ.{0,2}|.{0,2}משהב.{0,2}|.{0,2}משהכ.{0,2}|.{0,2}משהש.{0,2}|.{0,2}משהת.{0,2}|.{0,2}משהר.{0,2}|.{0,2}מושה.{0,2}|.{0,2}מישה.{0,2}|.{0,2}מאשה.{0,2}|.{0,2}מהשה.{0,2}|.{0,2}מפשה.{0,2}|.{0,2}מלשה.{0,2}|.{0,2}מנשה.{0,2}|.{0,2}מבשה.{0,2}|.{0,2}מכשה.{0,2}|.{0,2}מששה.{0,2})";

    #[test]
    fn whole_pattern_compiles_under_vendored_state_limit_and_split_succeeds() {
        // Baseline for the vendored tantivy-fst patch: this combined pattern
        // could not compile as one DFA under the upstream 1 000-state cap
        // (the bug the per-branch split fixed). Under the vendored 8 192-state
        // cap it compiles again — the fact that unlocks the relaxed phrase
        // budgets, since a phrase word is compiled joined.
        assert!(
            tantivy_fst::Regex::new(BOTH_OPTIONS_PATTERN).is_ok(),
            "combined 48-branch pattern no longer compiles — did the vendored \
             tantivy-fst STATE_LIMIT patch get lost?"
        );
        // The split path answers the query either way.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויאמר משה אל העם", "/books/a.txt");
        add(&mut engine, 2, "ספר בראשית", "/books/b.txt");
        engine.commit().unwrap();
        assert_eq!(search_ids(&mut engine, BOTH_OPTIONS_PATTERN), vec![1]);
    }

    #[test]
    fn both_options_pattern_now_returns_results() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויאמר משה אל העם", "/books/a.txt");
        add(&mut engine, 2, "ספר בראשית", "/books/b.txt");
        engine.commit().unwrap();

        // The exact pattern built for `משה` with "חלק ממילה" + "שגיאות כתיב".
        // As a single regex it exceeded the upstream 1000-state limit; the
        // per-branch split must find the document regardless of the cap.
        assert_eq!(search_ids(&mut engine, BOTH_OPTIONS_PATTERN), vec![1]);
    }

    #[test]
    fn single_alternation_matches_same_as_combined_regex() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "משה", "/books/a.txt");
        add(&mut engine, 2, "מסה", "/books/b.txt");
        add(&mut engine, 3, "בראשית", "/books/c.txt");
        engine.commit().unwrap();

        // OR of two branches — finds both documents, not the third.
        let mut ids = search_ids(&mut engine, "(משה|מסה)");
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn split_matching_parity_with_whole_pattern() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "משה", "/books/a.txt");
        add(&mut engine, 2, "מסה", "/books/b.txt");
        add(&mut engine, 3, "בראשית", "/books/c.txt");
        engine.commit().unwrap();

        // Capturing and non-capturing wrappers split identically (R1).
        let mut capturing = search_ids(&mut engine, "(משה|מסה)");
        capturing.sort_unstable();
        let mut non_capturing = search_ids(&mut engine, "(?:משה|מסה)");
        non_capturing.sort_unstable();
        assert_eq!(capturing, vec![1, 2]);
        assert_eq!(capturing, non_capturing);

        // A leading empty branch (all-optional-letter word, R7) contributes
        // nothing: no indexed term is empty.
        assert_eq!(search_ids(&mut engine, "(?:|משה)"), vec![1]);

        // A nested (non-top-level) alternation still compiles whole and
        // matches the same set.
        let mut nested = search_ids(&mut engine, ".{0,1}(משה|מסה).{0,1}");
        nested.sort_unstable();
        assert_eq!(nested, vec![1, 2]);
    }

    #[test]
    fn char_class_and_escape_branches_match_end_to_end() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספרים", "/books/a.txt");
        add(&mut engine, 2, "אב", "/books/b.txt");
        add(&mut engine, 3, "shalom", "/books/c.txt");
        engine.commit().unwrap();

        // Top-level alternation between two groups, with `|` inside a char
        // class in the second branch — the class pipe must not be split on.
        let mut ids = search_ids(&mut engine, "([א-ת]{2,4}(ים|ות|ה)?)|([א-ת]+[יו][ם|ן])");
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2]);

        // A bare char class containing `|` is a single literal pattern.
        assert_eq!(search_ids(&mut engine, "[ם|ן]"), Vec::<u64>::new());
    }

    #[test]
    fn global_max_expansions_enforced_across_all_branches() {
        let (mut engine, _dir) = make_engine();
        // Terms matched by *different* branches, so only the shared union can
        // cross the cap — no single branch does.
        add(&mut engine, 1, "אאא", "/books/a.txt");
        add(&mut engine, 2, "בבב", "/books/b.txt");
        add(&mut engine, 3, "גגג", "/books/c.txt");
        engine.commit().unwrap();

        let run = |engine: &mut SearchEngine, max_expansions: u32| {
            engine.search(
                vec!["(אאא|בבב|גגג)".to_string()],
                vec!["/root".to_string()],
                100,
                0,
                0,
                max_expansions,
                ResultsOrder::Catalogue,
                None,
            )
        };
        // Union of 3 terms exceeds a cap of 2 — the single-word path degrades
        // instead of erroring, keeping the first branches' terms (branch
        // order is the priority contract).
        let ids: Vec<u64> = run(&mut engine, 2)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec![1, 2]);
        // A cap of 3 fits the whole union.
        let ids: Vec<u64> = run(&mut engine, 3)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn advanced_typo_partial_single_word_finds_document() {
        // End-to-end through the app path (search_advanced): the exact option
        // combination that used to blow the 1000-state DFA cap and silently
        // return nothing. Runs on the relaxed single-word budget.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויאמר משה אל העם", "/books/a.txt");
        add(&mut engine, 2, "ספר בראשית", "/books/b.txt");
        engine.commit().unwrap();

        let mut options: HashMap<String, HashMap<String, bool>> = HashMap::new();
        options.insert(
            "משה_0".to_string(),
            HashMap::from([
                ("שגיאות כתיב".to_string(), true),
                ("חלק ממילה".to_string(), true),
            ]),
        );
        let ids: Vec<u64> = search_advanced_default(
            &engine,
            "משה".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            options,
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn advanced_typo_only_single_word_covers_full_edit_distance() {
        // typo-only single word expands through a Levenshtein-1 automaton
        // scan (VARIATION_CEILING_RESEARCH §3.ג): full edit-distance-1
        // coverage, a superset of the historical literal variant list.
        // "ספגר" — an insertion of ג, which is NOT in INSERTION_LETTERS — was
        // invisible to the sampled variants and must now match; a distance-2
        // word must not.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר בראשית", "/books/a.txt");
        add(&mut engine, 2, "ספגר אחר", "/books/b.txt");
        add(&mut engine, 3, "תורה צוה", "/books/c.txt");
        engine.commit().unwrap();

        let mut options: HashMap<String, HashMap<String, bool>> = HashMap::new();
        options.insert(
            "ספר_0".to_string(),
            HashMap::from([("שגיאות כתיב".to_string(), true)]),
        );
        let mut ids: Vec<u64> = search_advanced_default(
            &engine,
            "ספר".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            options,
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn invalid_branch_pattern_surfaces_error() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "משה", "/books/a.txt");
        engine.commit().unwrap();

        // One malformed branch must fail the search loudly, not silently
        // return nothing.
        let result = engine.search(
            vec!["משה|[".to_string()],
            vec!["/root".to_string()],
            100,
            0,
            0,
            100,
            ResultsOrder::Catalogue,
            None,
        );
        assert!(result.is_err());
    }

    // ── Snippet phrase-highlight filtering (results view ↔ search parity) ─

    /// Counts highlight spans (the `<font color=red>` opening tag the default
    /// `HighlightConfig` emits and the app's snippet parser expects).
    fn highlight_count(text: &str) -> usize {
        text.matches("<font color=red>").count()
    }

    #[test]
    fn advanced_phrase_snippet_highlights_only_adjacent_occurrence() {
        // The reported bug: searching the phrase "משה ואהרן" (no spacing) must
        // NOT light up a lone "משה" that is not followed by "ואהרן". tantivy's
        // term-based SnippetGenerator paints all three occurrences; the phrase
        // filter keeps only the adjacent pair.
        let (mut engine, _dir) = make_engine();
        add(
            &mut engine,
            1,
            "משה ואהרן אמרו שלום ואחר כך משה כהן הלך",
            "/books/a.txt",
        );
        engine.commit().unwrap();

        let results = search_advanced_default(
            &engine,
            "משה ואהרן".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        let text = &results[0].text;
        // Exactly the two words of the one adjacent phrase — not the lone משה.
        assert_eq!(highlight_count(text), 2, "snippet: {text}");
        assert!(text.contains("<font color=red>משה</font>"));
        assert!(text.contains("<font color=red>ואהרן</font>"));
        // The stray occurrence stays plain.
        assert!(text.contains("כך משה כהן"), "snippet: {text}");
    }

    #[test]
    fn advanced_single_word_snippet_still_highlights_every_occurrence() {
        // A single word is not a phrase: every occurrence is a real hit and must
        // stay highlighted — the filter only constrains multi-word phrases.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "משה דיבר ואחר כך משה שתק", "/books/a.txt");
        engine.commit().unwrap();

        let results = search_advanced_default(
            &engine,
            "משה".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            highlight_count(&results[0].text),
            2,
            "snippet: {}",
            results[0].text
        );
    }

    #[test]
    fn exact_phrase_snippet_highlights_only_adjacent_occurrence() {
        // The exact (PhraseQuery) path has the same term-based over-painting; a
        // strict-adjacency (all-zero gaps) filter drops the lone occurrence too.
        let (mut engine, _dir) = make_engine();
        add(
            &mut engine,
            1,
            "משה ואהרן אמרו שלום ואחר כך משה כהן הלך",
            "/books/a.txt",
        );
        engine.commit().unwrap();

        let results = engine
            .search_exact(
                "משה ואהרן".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        let text = &results[0].text;
        assert_eq!(highlight_count(text), 2, "snippet: {text}");
        assert!(text.contains("כך משה כהן"), "snippet: {text}");
    }

    #[test]
    fn advanced_phrase_custom_spacing_gap_is_highlighted_not_dropped() {
        // `distance` is 0 but `custom_spacing` permits one intermediate word, so
        // the search matches "משה <word> ואהרן". The filter's gap allowance must
        // come from the resolved gaps (= the custom spacing), NOT the raw
        // `distance`: with `distance` it would find no occurrence, fall back to
        // the plain term highlight, and re-paint the lone trailing "משה" (3
        // spans). With the gaps it paints exactly the gapped phrase (2 spans).
        let (mut engine, _dir) = make_engine();
        add(
            &mut engine,
            1,
            "משה רבנו ואהרן אמר ואחר כך משה לבדו הלך",
            "/books/a.txt",
        );
        engine.commit().unwrap();

        let results = search_advanced_default(
            &engine,
            "משה ואהרן".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,                                                     // distance 0 …
            HashMap::from([("0-1".to_string(), "1".to_string())]), // … but spacing allows 1
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        let text = &results[0].text;
        assert_eq!(highlight_count(text), 2, "snippet: {text}");
        assert!(text.contains("<font color=red>משה</font>"));
        assert!(text.contains("<font color=red>ואהרן</font>"));
        // The stray trailing occurrence stays plain.
        assert!(text.contains("כך משה לבדו"), "snippet: {text}");
    }

    // ── Phrase expansion degrade path (TermListPhraseQuery) ───────────────

    /// Ten distinct suffix letters so wildcard patterns fan out to ten index
    /// terms per position without leaving the Hebrew tokenizer's alphabet.
    const FAN_LETTERS: [&str; 10] = ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י"];

    fn make_fanout_phrase_engine() -> (SearchEngine, TempDir) {
        let (mut engine, dir) = make_engine();
        for (i, letter) in FAN_LETTERS.iter().enumerate() {
            add(
                &mut engine,
                i as u64 + 1,
                &format!("עמוד{letter} שער{letter}"),
                "/books/a.txt",
            );
        }
        // Reversed order — the phrase must reject it on every path.
        add(&mut engine, 50, "שערא עמודא", "/books/b.txt");
        // One intermediate word — admitted only when slop allows it.
        add(&mut engine, 60, "עמודב מלה שערב", "/books/c.txt");
        engine.commit().unwrap();
        (engine, dir)
    }

    fn phrase_ids(engine: &SearchEngine, slop: u32, max_expansions: u32) -> Vec<u64> {
        let mut ids: Vec<u64> = engine
            .search(
                vec!["עמוד.*".to_string(), "שער.*".to_string()],
                vec!["/root".to_string()],
                100,
                0,
                slop,
                max_expansions,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn phrase_over_expansion_ceiling_degrades_instead_of_erroring() {
        // Each position matches 10 index terms, so max_expansions = 5 puts
        // the query over tantivy's cumulative ceiling — historically a hard
        // `InvalidArgument("Phrase query exceeded max expansions")`. The
        // degrade path must serve the complete result set instead, with the
        // same in-order adjacency semantics as the exact path.
        let (engine, _dir) = make_fanout_phrase_engine();

        let ids = phrase_ids(&engine, 0, 5);
        assert_eq!(ids, (1..=10).collect::<Vec<u64>>());

        // Only the cumulative ceiling was outgrown — no per-position budget
        // truncated, so the count must not report partial results.
        let status = engine
            .count_with_status(
                vec!["עמוד.*".to_string(), "שער.*".to_string()],
                &["/root".to_string()],
                0,
                5,
            )
            .unwrap();
        assert_eq!(status.count, 10);
        assert!(!status.truncated);
    }

    #[test]
    fn phrase_degrade_path_matches_exact_path_results() {
        // The same query under a roomy ceiling runs the historical
        // RegexPhraseQuery engine; both engines must agree on every doc,
        // at slop 0 and with an allowance.
        let (engine, _dir) = make_fanout_phrase_engine();

        assert_eq!(phrase_ids(&engine, 0, 5), phrase_ids(&engine, 0, 1_000));
        let with_gap = phrase_ids(&engine, 1, 5);
        assert!(with_gap.contains(&60), "gap doc admitted under slop 1");
        assert_eq!(with_gap, phrase_ids(&engine, 1, 1_000));
    }

    #[test]
    fn phrase_expansion_limit_is_per_segment_not_global_union() {
        // Tantivy resets the cumulative phrase-expansion counter for every
        // segment. Each segment below has four expansions (three `עמוד.*`
        // terms plus one `שער.*` term), so max=5 is valid even though the
        // global union contains seven distinct terms. Counting the union
        // would wrongly select the flat-scoring degrade path.
        let (mut engine, _dir) = make_engine();
        disable_auto_merge(&engine);
        for (id, letter) in [(1, "א"), (2, "ב"), (3, "ג")] {
            add(
                &mut engine,
                id,
                &format!("עמוד{letter} שער"),
                "/books/a.txt",
            );
        }
        engine.commit().unwrap();
        for (id, letter) in [(4, "ד"), (5, "ה"), (6, "ו")] {
            add(
                &mut engine,
                id,
                &format!("עמוד{letter} שער"),
                "/books/b.txt",
            );
        }
        engine.commit().unwrap();

        let text_field = engine.schema.get_field("text").unwrap();
        assert!(!engine
            .phrase_exceeds_max_expansions(
                &["עמוד.*".to_string(), "שער".to_string()],
                text_field,
                5,
            )
            .unwrap());
        assert_eq!(phrase_ids(&engine, 0, 5), (1..=6).collect::<Vec<u64>>());
    }

    #[test]
    fn phrase_degrade_path_keeps_word_order() {
        let (engine, _dir) = make_fanout_phrase_engine();
        // Even with a generous allowance the reversed doc must stay out on
        // the degrade path (the AND driver alone would admit it).
        let ids = phrase_ids(&engine, 2, 5);
        assert!(!ids.contains(&50), "reversed order must not match");
    }

    #[test]
    fn phrase_position_over_budget_truncates_with_flag() {
        // One position fanning out past PHRASE_POSITION_MAX_EXPANSIONS must
        // degrade (truncate from the back of the dictionary order) and
        // surface `truncated`, while the lexicographically-early target term
        // survives and keeps matching.
        let (mut engine, _dir) = make_engine();
        let letters = [
            "א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י", "כ", "ל", "מ", "נ", "ס", "ע", "פ",
            "צ", "ק", "ר", "ש", "ת",
        ];
        // 22³ = 10 648 distinct terms under the "מלה" prefix — comfortably
        // past the 8 192 per-position ceiling. Spread over a few docs to
        // keep each document small.
        let mut words: Vec<String> = Vec::with_capacity(letters.len().pow(3));
        for a in letters {
            for b in letters {
                for c in letters {
                    words.push(format!("מלה{a}{b}{c}"));
                }
            }
        }
        for (chunk_index, chunk) in words.chunks(1_000).enumerate() {
            add(
                &mut engine,
                1_000 + chunk_index as u64,
                &chunk.join(" "),
                "/books/filler.txt",
            );
        }
        add(&mut engine, 1, "ראשא מלהאאא", "/books/a.txt");
        engine.commit().unwrap();

        let status = engine
            .count_with_status(
                vec!["ראש.*".to_string(), "מלה.*".to_string()],
                &["/root".to_string()],
                0,
                // Force the fallback on every normal-sized segment; the
                // second position then exceeds its own 8,192-term material-
                // ization ceiling across the index and must surface partial
                // results. Passing 8,192 here would correctly stay exact
                // because Tantivy applies that limit per segment.
                100,
            )
            .unwrap();
        assert!(status.truncated, "per-position overflow must surface");
        assert_eq!(status.count, 1, "the surviving target phrase still matches");
    }

    // ── Per-pair gap enforcement (GapVerifiedPhraseQuery) ─────────────────

    #[test]
    fn advanced_custom_spacing_is_enforced_per_pair() {
        // spacing = {0-1: 2, 1-2: 0}: up to two words between ויאמר and אל,
        // but אל and משה must be adjacent. Historically the engine collapsed
        // this to one global slop, so doc 2 — where the allowance is "spent"
        // in the wrong gap — also matched.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויאמר ה' צבאות אל משה", "/books/a.txt");
        add(&mut engine, 2, "ויאמר אל העם ואל משה", "/books/b.txt");
        engine.commit().unwrap();

        let spacing = HashMap::from([
            ("0-1".to_string(), "2".to_string()),
            ("1-2".to_string(), "0".to_string()),
        ]);
        let ids: Vec<u64> = search_advanced_default(
            &engine,
            "ויאמר אל משה".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            spacing.clone(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
        assert_eq!(ids, vec![1]);

        // The counting path runs through the same verified query.
        let count = count_advanced_default(
            &engine,
            "ויאמר אל משה".to_string(),
            vec!["/root".to_string()],
            0,
            spacing,
            HashMap::new(),
            HashMap::new(),
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn advanced_distance_allows_full_gap_in_every_pair() {
        // distance = 2 means "up to two words between EACH adjacent pair".
        // tantivy's slop is a cumulative budget, so passing the raw distance
        // used to reject a match that uses its allowance in both gaps at once.
        let (mut engine, _dir) = make_engine();
        add(
            &mut engine,
            1,
            "ויאמר ה' צבאות אל בני ישראל משה",
            "/books/a.txt",
        );
        // One gap over the allowance must still be rejected.
        add(
            &mut engine,
            2,
            "ויאמר אל דוד המלך איש הבינים משה",
            "/books/b.txt",
        );
        engine.commit().unwrap();

        let ids: Vec<u64> = search_advanced_default(
            &engine,
            "ויאמר אל משה".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            2,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn advanced_phrase_requires_query_word_order() {
        // tantivy's sloppy phrase matches unordered (positions compare by
        // abs_diff); the gap verifier restores the in-order semantics every
        // comment, highlight filter, and display pattern already promise.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויאמר להם משה", "/books/a.txt");
        // Reversed within the slop budget: tantivy's phrase scorer accepts
        // it, only the gap verifier rejects it.
        add(&mut engine, 2, "אמר משה ויאמר", "/books/b.txt");
        engine.commit().unwrap();

        let ids: Vec<u64> = search_advanced_default(
            &engine,
            "ויאמר משה".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            2,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn advanced_per_pair_gap_verifies_across_repeated_words() {
        // The feasibility sweep must consider EVERY chain start, not just the
        // earliest: here the first "אל" satisfies pair 0 but only the second
        // one can reach "משה" within pair 1's allowance.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויאמר אל העם ושוב אל משה", "/books/a.txt");
        engine.commit().unwrap();

        let spacing = HashMap::from([
            ("0-1".to_string(), "3".to_string()),
            ("1-2".to_string(), "0".to_string()),
        ]);
        let ids: Vec<u64> = search_advanced_default(
            &engine,
            "ויאמר אל משה".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            spacing,
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn advanced_per_pair_spacing_composes_with_word_options() {
        // Per-pair enforcement must hold when a word expands through an
        // option (grammatical prefixes): "ולמשה" matches the expanded word
        // pattern, and the pair allowances still gate the match.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויאמר שוב אל ולמשה", "/books/a.txt");
        add(&mut engine, 2, "ויאמר אל העם ולמשה", "/books/b.txt");
        engine.commit().unwrap();

        let options: HashMap<String, HashMap<String, bool>> = HashMap::from([(
            "משה_2".to_string(),
            HashMap::from([("קידומות דקדוקיות".to_string(), true)]),
        )]);
        let spacing = HashMap::from([
            ("0-1".to_string(), "1".to_string()),
            ("1-2".to_string(), "0".to_string()),
        ]);
        let ids: Vec<u64> = search_advanced_default(
            &engine,
            "ויאמר אל משה".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            spacing,
            HashMap::new(),
            options,
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
        assert_eq!(ids, vec![1]);
    }

    // ── Index-aware display highlight (parity with search, R5) ───────────

    /// Charwise display form of a nikud-free term, as the Dart layer receives
    /// it (each Hebrew letter may carry attached marks in displayed text, and
    /// between letters optional geresh/gershayim — the quote-free twin token).
    fn charwise(term: &str) -> String {
        let mut out = String::new();
        for (i, c) in term.chars().enumerate() {
            if i > 0 {
                out.push_str(crate::display_highlight::OPTIONAL_QUOTES);
            }
            out.push(c);
            out.push_str(crate::display_highlight::ATTACHED_MARKS_CLASS);
        }
        out
    }

    fn typo_options(word: &str) -> HashMap<String, HashMap<String, bool>> {
        HashMap::from([(
            format!("{word}_0"),
            HashMap::from([("שגיאות כתיב".to_string(), true)]),
        )])
    }

    #[test]
    fn index_highlight_paints_typo_matched_term() {
        let (mut engine, _dir) = make_engine();
        // The document is findable only via the typo variant מסה — the
        // query-shape pattern (which knows only משה + spelling) would leave
        // it with no highlight at all.
        add(&mut engine, 1, "ויקח מסה גדולה", "/books/a.txt");
        engine.commit().unwrap();

        let hl = engine
            .generate_index_highlight_pattern(
                "משה".to_string(),
                0,
                HashMap::new(),
                HashMap::new(),
                typo_options("משה"),
            )
            .unwrap()
            .expect("pattern");
        // Only מסה exists in this index, so it is the single branch.
        assert_eq!(hl.combined_pattern, charwise("מסה"));
        assert_eq!(hl.word_boundary_eligible, vec![true]);
    }

    #[test]
    fn index_highlight_partial_word_uses_query_shape_substring() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "הספרים הקדושים", "/books/a.txt");
        add(&mut engine, 2, "ספר תורה", "/books/b.txt");
        engine.commit().unwrap();

        let options = HashMap::from([(
            "ספר_0".to_string(),
            HashMap::from([("חלק ממילה".to_string(), true)]),
        )]);
        let hl = engine
            .generate_index_highlight_pattern(
                "ספר".to_string(),
                0,
                HashMap::new(),
                HashMap::new(),
                options.clone(),
            )
            .unwrap()
            .expect("pattern");

        // חלק-ממילה מדגיש את התת-מחרוזת שהוקלדה (query-shape) עם כיסוי מלא —
        // לא את המילים המנוטות השלמות מהאינדקס (שחסומות בתקציב ומחמיצות
        // מילים גלויות). ראה display_highlight::build_display_highlight_from_terms.
        let query_shape = generate_highlight_pattern(
            "ספר".to_string(),
            0,
            HashMap::new(),
            HashMap::new(),
            options,
        )
        .expect("pattern");
        assert_eq!(hl.combined_pattern, query_shape.combined_pattern);
        assert_eq!(hl.word_boundary_eligible, vec![false]);
    }

    #[test]
    fn index_highlight_falls_back_to_query_shape_when_nothing_matches() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "בראשית ברא", "/books/a.txt");
        engine.commit().unwrap();

        let from_index = engine
            .generate_index_highlight_pattern(
                "משה".to_string(),
                0,
                HashMap::new(),
                HashMap::new(),
                typo_options("משה"),
            )
            .unwrap()
            .expect("pattern");
        let query_shape = generate_highlight_pattern(
            "משה".to_string(),
            0,
            HashMap::new(),
            HashMap::new(),
            typo_options("משה"),
        )
        .expect("pattern");
        // Nothing in the index matches → never worse than the pure-string
        // pattern the app used until now.
        assert_eq!(from_index.combined_pattern, query_shape.combined_pattern);
        assert_eq!(
            from_index.word_boundary_eligible,
            query_shape.word_boundary_eligible
        );
    }

    #[test]
    fn index_highlight_multi_word_paints_each_words_variants() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויאמר מסה אל העם", "/books/a.txt");
        engine.commit().unwrap();

        let options = HashMap::from([(
            "משה_1".to_string(),
            HashMap::from([("שגיאות כתיב".to_string(), true)]),
        )]);
        let hl = engine
            .generate_index_highlight_pattern(
                "ויאמר משה".to_string(),
                0,
                HashMap::new(),
                HashMap::new(),
                options,
            )
            .unwrap()
            .expect("pattern");

        assert_eq!(hl.word_patterns.len(), 2);
        assert_eq!(hl.word_patterns[0], charwise("ויאמר"));
        assert_eq!(hl.word_patterns[1], charwise("מסה"));
        // Combined phrase pattern chains both words.
        assert!(hl.combined_pattern.starts_with(&charwise("ויאמר")));
        assert!(hl.combined_pattern.ends_with(&charwise("מסה")));
    }

    #[test]
    fn index_fuzzy_highlight_paints_edit_distance_matches() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ויקח מסה גדולה", "/books/a.txt");
        engine.commit().unwrap();

        let hl = engine
            .generate_index_fuzzy_highlight_pattern("משה".to_string(), 1)
            .unwrap()
            .expect("pattern");
        // The edit-distance-1 match from the index plus the literal token.
        assert!(hl.combined_pattern.contains(&charwise("מסה")));
        assert!(hl.combined_pattern.contains(&charwise("משה")));

        // Distance above the FuzzyTermQuery limit is rejected loudly.
        assert!(engine
            .generate_index_fuzzy_highlight_pattern("משה".to_string(), 3)
            .is_err());
    }

    #[test]
    fn index_highlight_plain_query_matches_exact_token() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        engine.commit().unwrap();

        // No options at all (the plain path of prepare_advanced_query).
        let hl = engine
            .generate_index_highlight_pattern(
                "שלום".to_string(),
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            )
            .unwrap()
            .expect("pattern");
        assert_eq!(hl.combined_pattern, charwise("שלום"));
    }

    #[test]
    fn test_count_by_book_basic() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        add(&mut engine, 3, "שלום חבר", "/books/b.txt");
        engine.commit().unwrap();

        let counts = engine
            .count_by_book(vec!["שלום".to_string()], vec!["/root".to_string()], 0, 100)
            .unwrap();

        assert_eq!(counts.get("/books/a.txt").copied(), Some(2));
        assert_eq!(counts.get("/books/b.txt").copied(), Some(1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_count_by_book_empty_result() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        engine.commit().unwrap();

        let counts = engine
            .count_by_book(vec!["ביי".to_string()], vec!["/root".to_string()], 0, 100)
            .unwrap();

        assert!(counts.is_empty());
    }

    #[test]
    fn test_count_by_book_no_cross_contamination() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום ביי", "/books/b.txt");
        engine.commit().unwrap();

        let counts = engine
            .count_by_book(vec!["עולם".to_string()], vec!["/root".to_string()], 0, 100)
            .unwrap();

        assert_eq!(counts.get("/books/a.txt").copied(), Some(1));
        assert_eq!(counts.get("/books/b.txt"), None);
    }

    #[test]
    fn test_count_by_book_multi_segment() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        engine.commit().unwrap();

        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        add(&mut engine, 3, "שלום חבר", "/books/b.txt");
        engine.commit().unwrap();

        let counts = engine
            .count_by_book(vec!["שלום".to_string()], vec!["/root".to_string()], 0, 100)
            .unwrap();

        assert_eq!(counts.get("/books/a.txt").copied(), Some(2));
        assert_eq!(counts.get("/books/b.txt").copied(), Some(1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_count_documents_by_file_path_empty_index() {
        let (engine, _dir) = make_engine();
        assert!(engine.count_documents_by_file_path().unwrap().is_empty());
        assert!(engine.get_indexed_file_paths().unwrap().is_empty());
    }

    #[test]
    fn test_count_documents_by_file_path_basic() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        add(&mut engine, 3, "שלום חבר", "/books/b.txt");
        engine.commit().unwrap();

        let counts = engine.count_documents_by_file_path().unwrap();
        assert_eq!(counts.get("/books/a.txt").copied(), Some(2));
        assert_eq!(counts.get("/books/b.txt").copied(), Some(1));
        assert_eq!(counts.len(), 2);

        let mut paths = engine.get_indexed_file_paths().unwrap();
        paths.sort();
        assert_eq!(paths, vec!["/books/a.txt", "/books/b.txt"]);
    }

    #[test]
    fn test_count_documents_by_file_path_respects_deletes() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        add(&mut engine, 3, "שלום חבר", "/books/b.txt");
        engine.commit().unwrap();

        engine.delete_document_by_id(1).unwrap();
        engine.delete_document_by_id(3).unwrap();
        engine.commit().unwrap();

        let counts = engine.count_documents_by_file_path().unwrap();
        assert_eq!(counts.get("/books/a.txt").copied(), Some(1));
        assert_eq!(
            counts.get("/books/b.txt"),
            None,
            "a book whose documents were all deleted must not be reported"
        );

        let paths = engine.get_indexed_file_paths().unwrap();
        assert_eq!(paths, vec!["/books/a.txt"]);
    }

    #[test]
    fn test_count_documents_by_file_path_multi_segment() {
        let (mut engine, _dir) = make_engine();
        disable_auto_merge(&engine);

        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        engine.commit().unwrap();
        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        add(&mut engine, 3, "שלום חבר", "/books/b.txt");
        engine.commit().unwrap();

        let counts = engine.count_documents_by_file_path().unwrap();
        assert_eq!(counts.get("/books/a.txt").copied(), Some(2));
        assert_eq!(counts.get("/books/b.txt").copied(), Some(1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_count_documents_by_file_path_excludes_uncommitted() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        engine.commit().unwrap();
        add(&mut engine, 2, "שלום רב", "/books/b.txt"); // not committed

        let counts = engine.count_documents_by_file_path().unwrap();
        assert_eq!(counts.len(), 1);
        assert_eq!(counts.get("/books/a.txt").copied(), Some(1));
    }

    #[test]
    fn test_count_documents_by_file_path_from_reopened_index() {
        // The motivating scenario: a fresh engine instance opens a directory
        // that already contains an index, and reconstructs which books are
        // indexed from the index itself (no external state).
        let dir = TempDir::new().unwrap();
        {
            let mut engine = SearchEngine::new(dir.path().to_str().unwrap());
            add(&mut engine, 1, "שלום עולם", "/books/a.txt");
            add(&mut engine, 2, "שלום רב", "/books/a.txt");
            add(&mut engine, 3, "שלום חבר", "/books/b.txt");
            engine.commit().unwrap();
        }

        let reopened = SearchEngine::new(dir.path().to_str().unwrap());
        let counts = reopened.count_documents_by_file_path().unwrap();
        assert_eq!(counts.get("/books/a.txt").copied(), Some(2));
        assert_eq!(counts.get("/books/b.txt").copied(), Some(1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_delete_document_by_id() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        engine.commit().unwrap();

        assert_eq!(
            engine
                .count(vec!["שלום".to_string()], &["/root".to_string()], 0, 100)
                .unwrap(),
            2
        );

        engine.delete_document_by_id(1).unwrap();
        engine.commit().unwrap();

        assert_eq!(
            engine
                .count(vec!["שלום".to_string()], &["/root".to_string()], 0, 100)
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_upsert_document() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "טקסט ישן", "/books/a.txt");
        engine.commit().unwrap();

        engine
            .upsert_document(
                1,
                "title",
                "ref",
                "/root",
                "טקסט חדש",
                0,
                false,
                "/books/a.txt",
                None,
                None,
                None,
            )
            .unwrap();
        engine.commit().unwrap();

        // Should have only one doc with id=1
        assert_eq!(
            engine
                .count(vec!["טקסט".to_string()], &["/root".to_string()], 0, 100)
                .unwrap(),
            1
        );
        assert_eq!(
            engine
                .count(vec!["ישן".to_string()], &["/root".to_string()], 0, 100)
                .unwrap(),
            0
        );
        assert_eq!(
            engine
                .count(vec!["חדש".to_string()], &["/root".to_string()], 0, 100)
                .unwrap(),
            1
        );
    }

    #[test]
    fn compute_content_fingerprint_is_stable_and_never_zero() {
        let a = compute_content_fingerprint("בראשית ברא".to_string());
        let b = compute_content_fingerprint("בראשית ברא".to_string());
        let c = compute_content_fingerprint("בראשית ברה".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, 0);
        assert_ne!(compute_content_fingerprint(String::new()), 0);
    }

    #[test]
    fn compute_content_fingerprint_bytes_matches_string_form() {
        let text = "בראשית ברא אלהים";
        assert_eq!(
            compute_content_fingerprint_bytes(text.as_bytes().to_vec()),
            compute_content_fingerprint(text.to_string()),
        );
        // UTF-8 קטוע — הפענוח ה-lossy זהה בשני הצדדים.
        let mut broken = text.as_bytes().to_vec();
        broken.truncate(broken.len() - 1);
        assert_eq!(
            compute_content_fingerprint_bytes(broken.clone()),
            compute_content_fingerprint(String::from_utf8_lossy(&broken).into_owned()),
        );
    }

    fn fingerprint_doc(id: u64, text: &str, file_path: &str, hash: Option<u64>) -> DocumentInput {
        DocumentInput {
            id,
            title: "ספר".to_string(),
            reference: "ref".to_string(),
            topics: "/root".to_string(),
            text: text.to_string(),
            segment: 0,
            is_pdf: false,
            file_path: file_path.to_string(),
            content_hash: hash,
            text_hash: hash,
            text_vocalized: None,
            section_id: None,
            generation_order: None,
            extra_facets: None,
        }
    }

    #[test]
    fn get_book_fingerprints_returns_per_book_hash() {
        let (mut engine, _dir) = make_engine();
        let hash_a = compute_content_fingerprint("ספר א".to_string());
        let hash_b = compute_content_fingerprint("ספר ב".to_string());
        engine
            .add_documents_batch(vec![
                fingerprint_doc(1, "שורה ראשונה", "id:1", Some(hash_a)),
                fingerprint_doc(2, "שורה שנייה", "id:1", Some(hash_a)),
                fingerprint_doc(3, "טקסט אחר", "id:2", Some(hash_b)),
                // PDF-כמו: ללא טביעת אצבע — צריך להופיע כ-0.
                fingerprint_doc(4, "עמוד", "C:/books/a.pdf", None),
            ])
            .unwrap();
        engine.commit().unwrap();

        let fingerprints = engine.get_book_fingerprints().unwrap();
        assert_eq!(fingerprints.get("id:1"), Some(&hash_a));
        assert_eq!(fingerprints.get("id:2"), Some(&hash_b));
        assert_eq!(fingerprints.get("C:/books/a.pdf"), Some(&0));
    }

    #[test]
    fn get_book_fingerprints_reflects_reindex_and_delete() {
        let (mut engine, _dir) = make_engine();
        let old_hash = compute_content_fingerprint("ישן".to_string());
        let new_hash = compute_content_fingerprint("חדש".to_string());
        engine
            .add_documents_batch(vec![
                fingerprint_doc(1, "ישן", "id:1", Some(old_hash)),
                fingerprint_doc(2, "אחר", "id:2", Some(old_hash)),
            ])
            .unwrap();
        engine.commit().unwrap();

        // אינדוקס מחדש של ספר: מחיקה לפי כותרת קודם הייתה מוחקת את שניהם —
        // כאן מדמים דרך upsert לפי id של מסמכי הספר בלבד.
        engine
            .upsert_documents_batch(vec![fingerprint_doc(1, "חדש", "id:1", Some(new_hash))])
            .unwrap();
        engine.commit().unwrap();

        let fingerprints = engine.get_book_fingerprints().unwrap();
        assert_eq!(fingerprints.get("id:1"), Some(&new_hash));
        assert_eq!(fingerprints.get("id:2"), Some(&old_hash));

        engine.remove_documents_by_title("ספר").unwrap();
        engine.commit().unwrap();
        assert!(engine.get_book_fingerprints().unwrap().is_empty());
    }

    #[test]
    fn delete_documents_by_file_path_removes_only_that_book() {
        let (mut engine, _dir) = make_engine();
        engine
            .add_documents_batch(vec![
                fingerprint_doc(1, "שורה ראשונה", "uid:1", Some(3)),
                fingerprint_doc(2, "שורה שנייה", "uid:1", Some(3)),
                fingerprint_doc(3, "טקסט אחר", "id:2", Some(5)),
            ])
            .unwrap();
        engine.commit().unwrap();

        engine.delete_documents_by_file_path("uid:1").unwrap();
        engine.commit().unwrap();

        // כל מסמכי uid:1 נמחקו — הספר האחר (בעל אותה כותרת) לא נפגע.
        let counts = engine.count_documents_by_file_path().unwrap();
        assert_eq!(counts.get("uid:1"), None);
        assert_eq!(counts.get("id:2"), Some(&1));

        // הצורה הקבוצתית מוחקת כמה ספרים בקריאה אחת.
        engine
            .delete_documents_by_file_paths(vec!["id:2".to_string(), "uid:404".to_string()])
            .unwrap();
        engine.commit().unwrap();
        assert!(engine.count_documents_by_file_path().unwrap().is_empty());
    }

    #[test]
    fn get_book_fingerprints_conflicting_docs_collapse_to_zero() {
        let (mut engine, _dir) = make_engine();
        engine
            .add_documents_batch(vec![
                fingerprint_doc(1, "שורה", "id:1", Some(7)),
                fingerprint_doc(2, "שורה", "id:1", Some(9)),
            ])
            .unwrap();
        engine.commit().unwrap();

        let fingerprints = engine.get_book_fingerprints().unwrap();
        assert_eq!(fingerprints.get("id:1"), Some(&0));
    }

    #[test]
    fn get_book_text_fingerprint_matches_map_form_per_book() {
        // בדיקת דריפט לספר אחד אינה משלמת O(total documents): ה-API הממוקד
        // חייב להחזיר בדיוק את מה שצורת המפה מחזירה לאותו ספר.
        let (mut engine, _dir) = make_engine();
        let text_a = "<h1>ספר א</h1>\nשורה";
        let text_b = "<h1>ספר ב</h1>\nשורה אחרת";
        for (order, path, text) in [(1u32, "id:1", text_a), (2, "id:2", text_b)] {
            engine
                .add_text_book(
                    "ספר".to_string(),
                    "/root".to_string(),
                    path.to_string(),
                    order,
                    DEFAULT_GENERATION_ORDER,
                    text.to_string(),
                    None,
                )
                .unwrap();
        }
        engine.commit().unwrap();

        let map = engine.get_book_text_fingerprints().unwrap();
        for (path, text) in [("id:1", text_a), ("id:2", text_b)] {
            let single = engine.get_book_text_fingerprint(path.to_string()).unwrap();
            assert_eq!(single, map[path]);
            assert_eq!(single, compute_content_fingerprint(text.to_string()));
        }

        // ספר שאינו באינדקס — "לא ניתן לאימות", לא שגיאה.
        assert_eq!(
            engine
                .get_book_text_fingerprint("id:404".to_string())
                .unwrap(),
            0
        );
    }

    #[test]
    fn get_book_text_fingerprint_is_zero_for_conflicting_and_pdf_books() {
        let (mut engine, _dir) = make_engine();
        // שתי חתימות שונות לאותו filePath (אינדוקס חלקי) — לא ניתן לאימות.
        engine
            .add_documents_batch(vec![
                fingerprint_doc(1, "שורה א", "id:1", Some(11)),
                fingerprint_doc(2, "שורה ב", "id:1", Some(22)),
                // PDF-כמו: בלי חתימת טקסט.
                fingerprint_doc(3, "עמוד", "C:/books/a.pdf", None),
            ])
            .unwrap();
        engine.commit().unwrap();

        assert_eq!(
            engine
                .get_book_text_fingerprint("id:1".to_string())
                .unwrap(),
            0
        );
        assert_eq!(
            engine
                .get_book_text_fingerprint("C:/books/a.pdf".to_string())
                .unwrap(),
            0
        );
    }

    #[test]
    fn text_fingerprint_survives_catalogue_order_change() {
        // מהות ההפרדה בין החתימות: אינדוקס-מחדש בסדר קטלוגי אחר (ספר נוסף
        // לספרייה) משנה את הקנונית אך לא את חתימת הטקסט — כך אימות דריפט
        // תוכן אינו נפסל משינויי קטלוג (issue Otzaria#828).
        let text = "<h1>ספר</h1>\nשורה של תוכן";
        let (mut engine, _dir) = make_engine();
        let index = |engine: &mut SearchEngine, order: u32| {
            engine
                .add_text_book(
                    "ספר".to_string(),
                    "/root".to_string(),
                    "id:1".to_string(),
                    order,
                    DEFAULT_GENERATION_ORDER,
                    text.to_string(),
                    None,
                )
                .unwrap();
            engine.commit().unwrap();
        };

        index(&mut engine, 5);
        let canonical_before = engine.get_book_fingerprints().unwrap()["id:1"];
        let text_before = engine.get_book_text_fingerprints().unwrap()["id:1"];

        engine.delete_documents_by_file_path("id:1").unwrap();
        index(&mut engine, 6);
        let canonical_after = engine.get_book_fingerprints().unwrap()["id:1"];
        let text_after = engine.get_book_text_fingerprints().unwrap()["id:1"];

        assert_ne!(canonical_before, canonical_after);
        assert_eq!(text_before, text_after);
        assert_eq!(text_after, compute_content_fingerprint(text.to_string()));
    }

    #[test]
    fn test_rollback() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        engine.commit().unwrap();

        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        engine.rollback().unwrap();
        engine.commit().unwrap();

        // doc 2 should not be present
        assert_eq!(
            engine
                .count(vec!["שלום".to_string()], &["/root".to_string()], 0, 100)
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_get_document_count() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום", "/books/a.txt");
        add(&mut engine, 2, "עולם", "/books/b.txt");
        engine.commit().unwrap();
        assert_eq!(engine.get_document_count(), 2);
    }

    #[test]
    fn test_get_document_by_id_found() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 42, "תורה ומצוות", "/books/a.txt");
        engine.commit().unwrap();

        let result = engine.get_document_by_id(42).unwrap();
        assert!(result.is_some());
        let doc = result.unwrap();
        assert_eq!(doc.id, 42);
        assert_eq!(doc.text, "תורה ומצוות");
    }

    #[test]
    fn test_get_document_by_id_not_found() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום", "/books/a.txt");
        engine.commit().unwrap();

        let result = engine.get_document_by_id(999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_search_fuzzy() {
        let (mut engine, _dir) = make_engine();
        // "שלום" exact match; "שלם" is one edit away (deletion); "ביי" is unrelated
        add(&mut engine, 1, "שלום", "/books/a.txt");
        add(&mut engine, 2, "שלם", "/books/b.txt");
        add(&mut engine, 3, "ביי", "/books/c.txt");
        engine.commit().unwrap();

        // distance=0: only exact match
        let exact = engine
            .search_fuzzy_terms(
                vec!["שלום".to_string()],
                vec!["/root".to_string()],
                10,
                0,
                0,
                ResultsOrder::Relevance,
                None,
            )
            .unwrap();
        let exact_texts: Vec<&str> = exact.iter().map(|r| r.text.as_str()).collect();
        assert!(
            exact_texts.contains(&"<font color=red>שלום</font>"),
            "distance=0 must return the exact match, highlighted"
        );
        assert!(
            !exact_texts.iter().any(|t| t.contains("שלם")),
            "distance=0 must not return near-match"
        );

        // distance=1: must return both "שלום" and the near-match "שלם"
        let fuzzy = engine
            .search_fuzzy_terms(
                vec!["שלום".to_string()],
                vec!["/root".to_string()],
                10,
                0,
                1,
                ResultsOrder::Relevance,
                None,
            )
            .unwrap();
        let fuzzy_texts: Vec<&str> = fuzzy.iter().map(|r| r.text.as_str()).collect();
        assert!(
            fuzzy_texts.contains(&"<font color=red>שלום</font>"),
            "distance=1 must return exact match, highlighted"
        );
        assert!(
            fuzzy_texts.contains(&"<font color=red>שלם</font>"),
            "distance=1 must return near-match one edit away, highlighted"
        );
        assert!(
            !fuzzy_texts.iter().any(|t| t.contains("ביי")),
            "unrelated term must not appear"
        );
    }

    #[test]
    fn test_set_magic_dictionary_path_reports_validity() {
        let (mut engine, dir) = make_engine();
        assert!(!engine.has_magic_dictionary());
        // Missing file → false, no error, no dictionary loaded.
        assert!(!engine
            .set_magic_dictionary_path(dir.path().join("nope.db").to_str().unwrap().to_string()));
        assert!(!engine.has_magic_dictionary());
        // Valid lexical.db → true.
        let db = make_lexical_db(&dir);
        assert!(engine.set_magic_dictionary_path(db));
        assert!(engine.has_magic_dictionary());
    }

    #[test]
    fn test_lexical_fuzzy_finds_inflection_exact_does_not() {
        let (mut engine, dir) = make_engine();
        // Only the inflected form is indexed; the lemma "הלך" is 3 edits away,
        // and no other token is within fuzzy distance 2 of it.
        add(&mut engine, 1, "הלכתי", "/books/a.txt");
        add(&mut engine, 2, "מזרח", "/books/b.txt");
        engine.commit().unwrap();

        // Exact "הלך" must NOT leak into the inflected doc.
        let exact = engine
            .search_exact(
                "הלך".to_string(),
                vec!["/root".to_string()],
                10,
                0,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap();
        assert!(
            exact.is_empty(),
            "exact search must not match the inflection"
        );

        // Fuzzy WITHOUT dictionary: "הלך"→"הלכתי" is >2 edits, still no match.
        let fuzzy_plain = engine
            .search_fuzzy(
                "הלך".to_string(),
                vec!["/root".to_string()],
                10,
                0,
                2,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap();
        assert!(
            fuzzy_plain.is_empty(),
            "plain fuzzy cannot reach the inflection at distance 2, got: {:?}",
            fuzzy_plain
                .iter()
                .map(|r| r.text.as_str())
                .collect::<Vec<_>>()
        );

        // Fuzzy WITH dictionary: the lexical expansion injects "הלכתי" → match.
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));
        let fuzzy_lex = engine
            .search_fuzzy(
                "הלך".to_string(),
                vec!["/root".to_string()],
                10,
                0,
                2,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(fuzzy_lex.len(), 1, "lexical fuzzy must find the inflection");
        assert!(fuzzy_lex[0].text.contains("הלכתי"));

        // count_fuzzy must agree with search_fuzzy (same matching logic).
        let count = engine
            .count_fuzzy(
                "הלך".to_string(),
                vec!["/root".to_string()],
                2,
                false,
                false,
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_lexical_fuzzy_distance_zero_stays_exact() {
        let (mut engine, dir) = make_engine();
        add(&mut engine, 1, "הלכתי", "/books/a.txt");
        engine.commit().unwrap();
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));

        let fuzzy_zero = engine
            .search_fuzzy(
                "הלך".to_string(),
                vec!["/root".to_string()],
                10,
                0,
                0,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap();
        assert!(
            fuzzy_zero.is_empty(),
            "max_distance=0 must not inject lexical expansions"
        );

        let count = engine
            .count_fuzzy(
                "הלך".to_string(),
                vec!["/root".to_string()],
                0,
                false,
                false,
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    fn fuzzy_ids(
        engine: &mut SearchEngine,
        query: &str,
        max_distance: u8,
        order: ResultsOrder,
    ) -> Vec<u64> {
        engine
            .search_fuzzy(
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                max_distance,
                order,
                false,
                false,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect()
    }

    #[test]
    fn test_lexical_fuzzy_relevance_tiers_exact_morphology_fuzzy() {
        let (mut engine, dir) = make_engine();
        add(&mut engine, 10, "הלך", "/books/a.txt"); // exact query token
        add(&mut engine, 5, "הלכתי", "/books/b.txt"); // dictionary surface form (distance 3)
        add(&mut engine, 1, "הלכה", "/books/c.txt"); // bare edit-distance neighbour (distance 1)
        engine.commit().unwrap();
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));

        // Boosting must not change recall: all three are still matched.
        let count = engine
            .count_fuzzy(
                "הלך".to_string(),
                vec!["/root".to_string()],
                2,
                false,
                false,
            )
            .unwrap();
        assert_eq!(count, 3, "ranking boosts must not change the recall set");

        // Relevance now tiers them: exact > morphology > edit-distance.
        let by_relevance = fuzzy_ids(&mut engine, "הלך", 2, ResultsOrder::Relevance);
        assert_eq!(
            by_relevance,
            vec![10, 5, 1],
            "relevance must rank exact, then dictionary form, then fuzzy"
        );

        // Catalogue ignores score and stays ordered by the catalogue id.
        let by_catalogue = fuzzy_ids(&mut engine, "הלך", 2, ResultsOrder::Catalogue);
        assert_eq!(
            by_catalogue,
            vec![1, 5, 10],
            "catalogue order must be unaffected by ranking"
        );
    }

    #[test]
    fn test_lexical_fuzzy_multi_word_relevance_differs_from_catalogue() {
        // The multi-word path is a `RegexPhraseQuery`, which (unlike the flat
        // single-token automaton) already scores by phrase frequency — so
        // relevance ordering is meaningful there without extra boosting.
        let (mut engine, dir) = make_engine();
        add(&mut engine, 1, "הלך מזרח", "/books/a.txt"); // phrase once
        add(&mut engine, 2, "הלך מזרח הלך מזרח", "/books/b.txt"); // phrase twice
        engine.commit().unwrap();
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));

        let by_catalogue = fuzzy_ids(&mut engine, "הלך מזרח", 2, ResultsOrder::Catalogue);
        assert_eq!(by_catalogue, vec![1, 2], "catalogue follows id order");

        let by_relevance = fuzzy_ids(&mut engine, "הלך מזרח", 2, ResultsOrder::Relevance);
        assert_eq!(
            by_relevance,
            vec![2, 1],
            "relevance must place the higher-frequency phrase first"
        );
    }

    #[test]
    fn test_lexical_fuzzy_exact_floor_survives_common_term() {
        // A near-ubiquitous exact term has BM25 idf ≈ 0, so a purely
        // multiplicative boost would sink it below the flat lexical tier. The
        // constant floor must keep exact hits on top regardless of doc frequency.
        let (mut engine, dir) = make_engine();
        for id in 1..=100u64 {
            add(&mut engine, id, "הלך", "/books/a.txt"); // exact in ~99% of docs
        }
        add(&mut engine, 1000, "הלכתי", "/books/b.txt"); // lone dictionary form
        engine.commit().unwrap();
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));

        let by_relevance: Vec<u64> = engine
            .search_fuzzy(
                "הלך".to_string(),
                vec!["/root".to_string()],
                200,
                0,
                2,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(by_relevance.len(), 101, "all docs must be recalled");
        assert_eq!(
            by_relevance.last(),
            Some(&1000),
            "the lone lexical form must rank below every exact hit despite idf≈0"
        );
    }

    #[test]
    fn test_plain_fuzzy_relevance_ranks_exact_first() {
        // No dictionary loaded — exercises the plain fuzzy builder.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 9, "כתבה", "/books/a.txt"); // edit-distance neighbour (distance 1)
        add(&mut engine, 2, "כתב", "/books/b.txt"); // exact query token
        engine.commit().unwrap();

        let by_relevance = fuzzy_ids(&mut engine, "כתב", 2, ResultsOrder::Relevance);
        assert_eq!(
            by_relevance,
            vec![2, 9],
            "exact match must outrank a bare fuzzy neighbour"
        );

        // distance 0 stays a pure exact match — recall is byte-identical.
        let zero = fuzzy_ids(&mut engine, "כתב", 0, ResultsOrder::Relevance);
        assert_eq!(zero, vec![2], "distance 0 must match only the exact token");
    }

    #[test]
    fn test_lexical_fuzzy_multi_word_requires_phrase() {
        let (mut engine, dir) = make_engine();
        add(&mut engine, 1, "הלכתי לישון", "/books/a.txt");
        add(
            &mut engine,
            2,
            "הלכתי ואז דיברתי הרבה לפני לישון",
            "/books/b.txt",
        );
        add(&mut engine, 3, "לישון הלכתי", "/books/c.txt");
        add(&mut engine, 4, "הלכתי", "/books/d.txt");
        add(&mut engine, 5, "לכו ונכהו בלשון", "/books/e.txt");
        engine.commit().unwrap();
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));

        let got = ids(engine
            .search_fuzzy(
                "הלכתי לישון".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                2,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap());
        assert_eq!(
            got,
            vec![1, 5],
            "multi-token lexical fuzzy search should preserve order while allowing one intervening token"
        );

        let count = engine
            .count_fuzzy(
                "הלכתי לישון".to_string(),
                vec!["/root".to_string()],
                2,
                false,
                false,
            )
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_lexical_fuzzy_highlights_literal_second_token() {
        let (mut engine, dir) = make_engine();
        add(&mut engine, 1, "הולכים לישון", "/books/a.txt");
        for (idx, term) in one_edit_insertions("לישון").into_iter().enumerate() {
            add(
                &mut engine,
                idx as u64 + 2,
                &format!("רעש {term}"),
                "/books/noise.txt",
            );
        }
        engine.commit().unwrap();
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));

        let results = engine
            .search_fuzzy(
                "הלכתי לישון".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                2,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        let hit = results.iter().find(|result| result.id == 1).unwrap();
        assert!(
            hit.text.contains("<font color=red>לישון</font>"),
            "the literal second query token must remain highlighted, got: {}",
            hit.text
        );
    }

    #[test]
    fn test_lexical_fuzzy_multi_word_allows_expansions_per_token() {
        let (mut engine, dir) = make_engine();
        add(&mut engine, 1, "הלכתי לישון", "/books/a.txt");

        let mut next_id = 2u64;
        for term in one_edit_insertions("הלכתי")
            .into_iter()
            .chain(one_edit_insertions("לישון"))
        {
            add(&mut engine, next_id, &term, "/books/noise.txt");
            next_id += 1;
        }
        engine.commit().unwrap();
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));

        let got = ids(engine
            .search_fuzzy(
                "הלכתי לישון".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                1,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap());
        assert_eq!(got, vec![1]);
    }

    fn one_edit_insertions(token: &str) -> Vec<String> {
        const LETTERS: &[char] = &[
            'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ',
            'צ', 'ק', 'ר', 'ש', 'ת',
        ];

        let chars: Vec<char> = token.chars().collect();
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for position in 0..=chars.len() {
            for letter in LETTERS {
                let mut variant = chars.clone();
                variant.insert(position, *letter);
                let variant: String = variant.into_iter().collect();
                if variant != token && seen.insert(variant.clone()) {
                    out.push(variant);
                }
            }
        }
        out
    }

    #[test]
    fn test_new_tolerates_held_writer_lock() {
        let (mut first, dir) = make_engine();
        add(&mut first, 1, "ספר", "/books/a.txt");
        first.commit().unwrap();

        // While `first` holds the writer lock, a second engine must still open
        // (no panic) and serve reads.
        let mut second = SearchEngine::new(dir.path().to_str().unwrap());
        assert_eq!(search_ids(&mut second, "ספר"), vec![1]);

        // Once the lock is released, writes recover lazily via ensure_writer.
        drop(first);
        add(&mut second, 2, "תורה", "/books/b.txt");
        second.commit().unwrap();
        assert_eq!(search_ids(&mut second, "תורה"), vec![2]);
    }

    #[test]
    fn test_search_and_count() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        add(&mut engine, 3, "ביי", "/books/b.txt");
        engine.commit().unwrap();

        let page = engine
            .search_and_count(
                vec!["שלום".to_string()],
                vec!["/root".to_string()],
                1,
                0,
                0,
                100,
                ResultsOrder::Relevance,
                None,
            )
            .unwrap();

        assert_eq!(
            page.total_count, 2,
            "total_count should reflect all hits, not just page size"
        );
        assert_eq!(
            page.results.len(),
            1,
            "results should be limited by limit param"
        );
    }

    #[test]
    fn test_search_offset() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום רב", "/books/b.txt");
        add(&mut engine, 3, "שלום חבר", "/books/c.txt");
        engine.commit().unwrap();

        let page1 = engine
            .search(
                vec!["שלום".to_string()],
                vec!["/root".to_string()],
                2,
                0,
                0,
                100,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap();
        let page2 = engine
            .search(
                vec!["שלום".to_string()],
                vec!["/root".to_string()],
                2,
                2,
                0,
                100,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap();

        assert_eq!(page1.len(), 2);
        assert_eq!(page2.len(), 1);
        // Pages must not overlap
        let ids1: Vec<u64> = page1.iter().map(|r| r.id).collect();
        let ids2: Vec<u64> = page2.iter().map(|r| r.id).collect();
        assert!(ids1.iter().all(|id| !ids2.contains(id)));
    }

    #[test]
    fn test_optimize_reduces_segments_many_commits() {
        let (mut engine, _dir) = make_engine();
        disable_auto_merge(&engine);

        for id in 1..=12 {
            let text = format!("שלום {id}");
            let file_path = format!("/books/{id}.txt");
            add(&mut engine, id, &text, &file_path);
            engine.commit().unwrap();
        }

        let before = engine.get_segment_count().unwrap();
        assert!(before > 1, "test setup should create multiple segments");

        engine.optimize().unwrap();

        let after = engine.get_segment_count().unwrap();

        assert!(
            after <= before,
            "optimize should not increase segment count"
        );
        assert_eq!(
            after, MAX_SEGMENTS_AFTER_OPTIMIZE as u32,
            "optimize compacts the smallest segments down to the cap"
        );
        assert_eq!(engine.get_document_count(), 12);
    }

    #[test]
    fn test_optimize_commits_pending_documents() {
        let (mut engine, _dir) = make_engine();
        disable_auto_merge(&engine);
        // Two committed segments so optimize doesn't take the early-skip path.
        add(&mut engine, 1, "ספר", "/books/a.txt");
        engine.commit().unwrap();
        add(&mut engine, 2, "ספר", "/books/b.txt");
        engine.commit().unwrap();

        // A pending document must survive optimize, not vanish with the
        // discarded writer buffer.
        add(&mut engine, 3, "ספר", "/books/c.txt");
        engine.optimize().unwrap();

        assert_eq!(search_ids(&mut engine, "ספר"), vec![1, 2, 3]);
    }

    #[test]
    fn test_optimize_preserves_search_results() {
        let (mut engine, _dir) = make_engine();
        disable_auto_merge(&engine);

        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        engine.commit().unwrap();
        add(&mut engine, 2, "שלום רב", "/books/b.txt");
        engine.commit().unwrap();
        add(&mut engine, 3, "ביי", "/books/c.txt");
        engine.commit().unwrap();
        add(&mut engine, 4, "שלום חבר", "/books/d.txt");
        engine.commit().unwrap();

        let before_ids = search_ids(&mut engine, "שלום");
        engine.optimize().unwrap();
        let after_ids = search_ids(&mut engine, "שלום");

        assert_eq!(
            before_ids, after_ids,
            "optimize must preserve search results"
        );
    }

    #[test]
    fn test_optimize_preserves_upsert_and_delete_afterwards() {
        let (mut engine, _dir) = make_engine();
        disable_auto_merge(&engine);

        add(&mut engine, 1, "טקסט ישן", "/books/a.txt");
        engine.commit().unwrap();
        add(&mut engine, 2, "למחיקה", "/books/b.txt");
        engine.commit().unwrap();

        engine.optimize().unwrap();

        engine
            .upsert_document(
                1,
                "title",
                "ref",
                "/root",
                "טקסט חדש",
                0,
                false,
                "/books/a.txt",
                None,
                None,
                None,
            )
            .unwrap();
        engine.delete_document_by_id(2).unwrap();
        engine.commit().unwrap();

        assert_eq!(search_ids(&mut engine, "ישן"), Vec::<u64>::new());
        assert_eq!(search_ids(&mut engine, "חדש"), vec![1]);
        assert!(engine.get_document_by_id(2).unwrap().is_none());
    }

    #[test]
    fn test_optimize_noop_when_single_segment() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום", "/books/a.txt");
        engine.commit().unwrap();

        let before = engine.get_segment_count().unwrap();
        engine.optimize().unwrap();
        let after = engine.get_segment_count().unwrap();

        assert_eq!(before, 1);
        assert_eq!(after, 1);

        add(&mut engine, 2, "עולם", "/books/b.txt");
        engine.commit().unwrap();
        assert_eq!(search_ids(&mut engine, "עולם"), vec![2]);
    }

    #[test]
    fn test_writer_reopens_after_transient_reopen_failure() {
        let (mut engine, _dir) = make_engine();

        engine.index_writer = None;
        let competing_writer: IndexWriter<TantivyDocument> =
            engine.index.writer(DEFAULT_WRITER_HEAP_SIZE).unwrap();

        let err = engine
            .add_document(
                1,
                "title",
                "ref",
                "/root",
                "שלום",
                0,
                false,
                "/books/a.txt",
                None,
                None,
                None,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("Failed to acquire index lock")
                || err.to_string().contains("LockFailure"),
            "unexpected error: {err:#}"
        );
        assert!(engine.index_writer.is_none());

        drop(competing_writer);

        add(&mut engine, 1, "שלום", "/books/a.txt");
        engine.commit().unwrap();

        assert_eq!(search_ids(&mut engine, "שלום"), vec![1]);
    }

    #[test]
    fn test_clear_reopens_after_transient_reopen_failure() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום", "/books/a.txt");
        engine.commit().unwrap();

        engine.index_writer = None;
        let competing_writer: IndexWriter<TantivyDocument> =
            engine.index.writer(DEFAULT_WRITER_HEAP_SIZE).unwrap();

        let err = engine.clear().unwrap_err();
        assert!(
            err.to_string().contains("Failed to acquire index lock")
                || err.to_string().contains("LockFailure"),
            "unexpected error: {err:#}"
        );
        assert!(engine.index_writer.is_none());

        drop(competing_writer);

        engine.clear().unwrap();
        engine.commit().unwrap();

        assert_eq!(engine.get_document_count(), 0);
        assert_eq!(search_ids(&mut engine, "שלום"), Vec::<u64>::new());
    }

    // ── High-level mode-specific API ─────────────────────────────────────────────

    fn ids(results: Vec<SearchResult>) -> Vec<u64> {
        let mut v: Vec<u64> = results.into_iter().map(|r| r.id).collect();
        v.sort();
        v
    }

    #[test]
    fn test_search_exact_single_and_phrase() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום רב", "/books/b.txt");
        engine.commit().unwrap();

        // Single token matches both docs containing the word.
        let got = ids(engine
            .search_exact(
                "שלום".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap());
        assert_eq!(got, vec![1, 2]);

        // Phrase matches only the doc with those adjacent words.
        let got = ids(engine
            .search_exact(
                "שלום עולם".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap());
        assert_eq!(got, vec![1]);
    }

    #[test]
    fn test_search_exact_strips_query_nikud() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום", "/books/a.txt"); // indexed without nikud
        engine.commit().unwrap();

        // Query carries nikud; exact mode strips it before tokenizing.
        let got = ids(engine
            .search_exact(
                "שָׁלוֹם".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap());
        assert_eq!(got, vec![1]);
    }

    #[test]
    fn test_stored_text_keeps_punctuation_and_highlight_lands_on_it() {
        let (mut engine, _dir) = make_engine();
        // מקצה-לקצה של issue #446/#500: הטקסט השמור משמר פיסוק ונקי מניקוד
        // ומ-Presentation Forms, ושאילתה רגילה מוצאת ומדגישה אותו.
        let raw = "וּבָזֶה יוּבַן שַׁ\"ס (עא:) הַמְמַלֵּא גְּרוֹנָם, מ\u{FB1D}ם וכו'";
        let stored = crate::hebrew_query::normalize_text_for_indexing(raw);
        assert_eq!(stored, "ובזה יובן ש\"ס (עא:) הממלא גרונם, מים וכו'");
        engine
            .add_document(
                1,
                "title",
                "ref",
                "/root",
                &stored,
                0,
                false,
                "/books/a.txt",
                None,
                None,
                None,
            )
            .unwrap();
        engine.commit().unwrap();

        for query in ["הממלא", "מים", "עא"] {
            let results = engine
                .search_exact(
                    query.to_string(),
                    vec!["/root".to_string()],
                    100,
                    0,
                    ResultsOrder::Catalogue,
                    false,
                    false,
                    None,
                )
                .unwrap();
            assert_eq!(ids(results.clone()), vec![1], "no hit for {query}");
            assert!(
                results[0]
                    .text
                    .contains(&format!("<font color=red>{query}</font>")),
                "highlight missing for {query}: {}",
                results[0].text
            );
        }
    }

    #[test]
    fn test_search_advanced_grammatical_prefix() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        add(&mut engine, 2, "הספר", "/books/b.txt");
        add(&mut engine, 3, "מטבע", "/books/c.txt");
        engine.commit().unwrap();

        let mut word_opts = HashMap::new();
        word_opts.insert("קידומות דקדוקיות".to_string(), true);
        let mut options = HashMap::new();
        options.insert("ספר_0".to_string(), word_opts);

        let got = ids(search_advanced_default(
            &engine,
            "ספר".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            options,
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap());
        assert_eq!(
            got,
            vec![1, 2],
            "grammatical prefix should match ספר and הספר"
        );
    }

    #[test]
    fn test_search_advanced_heavy_option_combo_compiles_and_runs() {
        // typo + grammatical prefix + grammatical suffix produces the largest
        // morphological regex. The length budget must keep it under tantivy-fst's
        // DFA state limit so the search compiles and returns instead of erroring.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        add(&mut engine, 2, "הספרים", "/books/b.txt");
        engine.commit().unwrap();

        let mut word_opts = HashMap::new();
        word_opts.insert(hebrew_query::OPT_TYPO.to_string(), true);
        word_opts.insert("קידומות דקדוקיות".to_string(), true);
        word_opts.insert("סיומות דקדוקיות".to_string(), true);
        let mut options = HashMap::new();
        options.insert("ספר_0".to_string(), word_opts);

        let result = search_advanced_default(
            &engine,
            "ספר".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            options,
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        );
        let got = ids(result.expect("heavy option combo must compile and run"));
        assert!(
            got.contains(&1),
            "the base word ספר should still match, got {got:?}"
        );
    }

    #[test]
    fn test_single_term_respects_max_expansions() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        add(&mut engine, 2, "הספר", "/books/b.txt");
        engine.commit().unwrap();

        // Two index terms match; a cap of 1 truncates the term set (degrade,
        // not error) — exactly one document comes back. Which term survives
        // depends on segment order (each doc may land in its own segment),
        // so only the count is pinned.
        let truncated = ids(engine
            .search(
                vec![".*ספר".to_string()],
                vec![],
                100,
                0,
                0,
                1,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap());
        assert_eq!(
            truncated.len(),
            1,
            "a cap of 1 should serve exactly one term's documents, got {truncated:?}"
        );
        assert!(
            truncated[0] == 1 || truncated[0] == 2,
            "unexpected document {truncated:?}"
        );

        let ok = ids(engine
            .search(
                vec![".*ספר".to_string()],
                vec![],
                100,
                0,
                0,
                10,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap());
        assert_eq!(ok, vec![1, 2]);
    }

    #[test]
    fn test_single_term_truncation_flag_surfaces() {
        // The degrade path must report itself so the stream can flag partial
        // results to the UI, instead of silently serving a subset.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        add(&mut engine, 2, "הספר", "/books/b.txt");
        engine.commit().unwrap();

        // Two terms match `.*ספר`; a cap of 1 stops collection early.
        let (_, truncated) = engine
            .build_query(vec![".*ספר".to_string()], vec![], 0, 1)
            .unwrap();
        assert!(
            truncated,
            "a cap of 1 over two matching terms must report truncation"
        );

        // A generous cap collects everything — no degrade, no flag.
        let (_, not_truncated) = engine
            .build_query(vec![".*ספר".to_string()], vec![], 0, 100)
            .unwrap();
        assert!(
            !not_truncated,
            "a cap that fits every term must not report truncation"
        );
    }

    #[test]
    fn test_search_and_count_propagates_truncation_flag() {
        // The page/count API must surface the same degrade signal the stream
        // does — dropping it lets a consumer show partial results unwarned.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        add(&mut engine, 2, "הספר", "/books/b.txt");
        engine.commit().unwrap();

        let page = engine
            .search_and_count(
                vec![".*ספר".to_string()],
                vec![],
                100,
                0,
                0,
                1,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap();
        assert!(
            page.truncated,
            "a cap of 1 over two matching terms must flag the page result"
        );

        let page = engine
            .search_and_count(
                vec![".*ספר".to_string()],
                vec![],
                100,
                0,
                0,
                100,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap();
        assert!(!page.truncated, "an uncapped query must not flag the page");
        assert_eq!(page.total_count, 2);
    }

    #[test]
    fn test_count_apis_with_status_surface_truncation() {
        // count/count_by_book/get_facet_counts feed the facet filter tree; the
        // *_with_status variants must carry the same degrade signal so a
        // consumer can flag partial counts instead of showing them as exact.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        add(&mut engine, 2, "הספר", "/books/b.txt");
        engine.commit().unwrap();

        let capped = engine
            .count_with_status(vec![".*ספר".to_string()], &[], 0, 1)
            .unwrap();
        assert!(capped.truncated, "a cap of 1 must flag count_with_status");

        let books = engine
            .count_by_book_with_status(vec![".*ספר".to_string()], vec![], 0, 1)
            .unwrap();
        assert!(books.truncated, "a cap of 1 must flag the per-book counts");

        let facets = engine
            .get_facet_counts_with_status(vec![".*ספר".to_string()], vec![], "/".to_string(), 0, 1)
            .unwrap();
        assert!(facets.truncated, "a cap of 1 must flag the facet counts");

        let uncapped = engine
            .count_with_status(vec![".*ספר".to_string()], &[], 0, 100)
            .unwrap();
        assert!(!uncapped.truncated, "an uncapped count must not flag");
        assert_eq!(uncapped.count, 2);

        let facets = engine
            .get_facet_counts_with_status(
                vec![".*ספר".to_string()],
                vec![],
                "/".to_string(),
                0,
                100,
            )
            .unwrap();
        assert!(!facets.truncated, "an uncapped facet count must not flag");
        assert!(
            facets
                .counts
                .iter()
                .any(|f| f.path == "/root" && f.count == 2),
            "facet counts should be exact when not truncated"
        );

        // The bare API stays backward compatible: same count, flag dropped.
        let bare = engine
            .count(vec![".*ספר".to_string()], &[], 0, 100)
            .unwrap();
        assert_eq!(bare, uncapped.count);
    }

    #[test]
    fn test_advanced_count_apis_with_status_surface_truncation() {
        // The advanced *_with_status trio must propagate the degrade signal
        // through build_advanced_query. "חלק ממילה" on a one-letter word gets
        // the relaxed 20 000-term cap (plain_max_expansions), so an index
        // with 20 812 matching terms overflows it.
        let (mut engine, _dir) = make_engine();
        let letters = [
            'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ',
            'צ', 'ק', 'ר', 'ש', 'ת',
        ];
        // 22³ words starting with א plus 21×22² with א second — all distinct,
        // all inside the `.{0,3}א.{0,3}` partial window.
        let mut words: Vec<String> = Vec::new();
        for a in letters {
            for b in letters {
                for c in letters {
                    words.push(format!("א{a}{b}{c}"));
                    if a != 'א' {
                        words.push(format!("{a}א{b}{c}"));
                    }
                }
            }
        }
        for (i, chunk) in words.chunks(4_000).enumerate() {
            add(&mut engine, i as u64 + 1, &chunk.join(" "), "/books/a.txt");
        }
        // Control docs for the under-cap path (no א anywhere).
        add(&mut engine, 100, "שלום עולם", "/books/b.txt");
        add(&mut engine, 101, "שלום", "/books/c.txt");
        engine.commit().unwrap();

        let partial_on = |word: &str| -> HashMap<String, HashMap<String, bool>> {
            HashMap::from([(
                format!("{word}_0"),
                HashMap::from([("חלק ממילה".to_string(), true)]),
            )])
        };

        let count = engine
            .count_advanced_with_status(
                "א".to_string(),
                String::new(),
                vec!["/root".to_string()],
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                partial_on("א"),
                HashMap::new(),
                false,
                false,
                SearchScope::WordDistance,
                SearchScope::WordDistance,
                None,
                None,
            )
            .unwrap();
        assert!(
            count.truncated,
            "20 812 matching terms over the 20 000 cap must flag the advanced count"
        );

        let books = engine
            .count_by_book_advanced_with_status(
                "א".to_string(),
                String::new(),
                vec!["/root".to_string()],
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                partial_on("א"),
                HashMap::new(),
                false,
                false,
                SearchScope::WordDistance,
                SearchScope::WordDistance,
                None,
                None,
            )
            .unwrap();
        assert!(
            books.truncated,
            "the advanced per-book counts must carry the same flag"
        );

        let facets = engine
            .get_facet_counts_advanced_with_status(
                "א".to_string(),
                String::new(),
                vec!["/root".to_string()],
                "/".to_string(),
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                partial_on("א"),
                HashMap::new(),
                false,
                false,
                SearchScope::WordDistance,
                SearchScope::WordDistance,
                None,
                None,
            )
            .unwrap();
        assert!(
            facets.truncated,
            "the advanced facet counts must carry the same flag"
        );

        // A word matching a single term stays far under the cap: no flag,
        // exact counts on every path.
        let count = engine
            .count_advanced_with_status(
                "שלום".to_string(),
                String::new(),
                vec!["/root".to_string()],
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                partial_on("שלום"),
                HashMap::new(),
                false,
                false,
                SearchScope::WordDistance,
                SearchScope::WordDistance,
                None,
                None,
            )
            .unwrap();
        assert!(!count.truncated, "under the cap the count must not flag");
        assert_eq!(count.count, 2, "both control documents match");

        let facets = engine
            .get_facet_counts_advanced_with_status(
                "שלום".to_string(),
                String::new(),
                vec!["/root".to_string()],
                "/".to_string(),
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                partial_on("שלום"),
                HashMap::new(),
                false,
                false,
                SearchScope::WordDistance,
                SearchScope::WordDistance,
                None,
                None,
            )
            .unwrap();
        assert!(!facets.truncated, "under the cap the facets must not flag");
        assert!(
            facets
                .counts
                .iter()
                .any(|f| f.path == "/root" && f.count == 2),
            "facet counts should be exact when not truncated"
        );
    }

    #[test]
    fn test_search_advanced_strips_query_nikud() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר תורה", "/books/a.txt");
        engine.commit().unwrap();

        // Pasted vocalized text must still match the nikud-free index terms.
        let got = ids(search_advanced_default(
            &engine,
            "סֵפֶר".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap());
        assert_eq!(got, vec![1], "vocalized advanced query should match");
    }

    #[test]
    fn test_search_advanced_empty_query_returns_no_results() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        engine.commit().unwrap();

        // Empty and punctuation-only queries produce zero regex terms; they must
        // return no results instead of panicking inside RegexPhraseQuery.
        for query in ["", "?!"] {
            let results = search_advanced_default(
                &engine,
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap();
            assert!(results.is_empty(), "query {query:?} should match nothing");
        }
    }

    #[test]
    fn test_search_skips_empty_facets() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        engine.commit().unwrap();

        let got = ids(engine
            .search(
                vec!["ספר".to_string()],
                vec![],
                100,
                0,
                0,
                100,
                ResultsOrder::Catalogue,
                None,
            )
            .unwrap());
        assert_eq!(got, vec![1], "empty facet list should not filter anything");
    }

    #[test]
    fn test_search_rejects_invalid_facet() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        engine.commit().unwrap();

        let result = engine.search(
            vec!["ספר".to_string()],
            vec!["not-a-facet".to_string()],
            100,
            0,
            0,
            100,
            ResultsOrder::Catalogue,
            None,
        );
        assert!(result.is_err(), "malformed facet should error, not panic");
    }

    #[test]
    fn test_search_advanced_highlights_morphological_variant() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "ספר", "/books/a.txt");
        add(&mut engine, 2, "הספר", "/books/b.txt");
        engine.commit().unwrap();

        let mut word_opts = HashMap::new();
        word_opts.insert("קידומות דקדוקיות".to_string(), true);
        let mut options = HashMap::new();
        options.insert("ספר_0".to_string(), word_opts);

        let results = search_advanced_default(
            &engine,
            "ספר".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            options,
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap();

        // The query matched the prefixed variant "הספר" via regex; highlighting
        // must wrap the variant that actually matched, not just the literal "ספר".
        let variant = results
            .iter()
            .find(|r| r.id == 2)
            .expect("הספר document should be in results");
        assert_eq!(
            variant.text, "<font color=red>הספר</font>",
            "morphological variant should be highlighted"
        );
    }

    #[test]
    fn test_search_advanced_alternative_words() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "מלך", "/books/a.txt");
        add(&mut engine, 2, "שר", "/books/b.txt");
        add(&mut engine, 3, "עיר", "/books/c.txt");
        engine.commit().unwrap();

        let mut alts = HashMap::new();
        alts.insert(0u32, vec!["מלך".to_string()]);
        let got = ids(search_advanced_default(
            &engine,
            "שר".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            alts,
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap());
        assert_eq!(got, vec![1, 2], "alternatives should OR שר with מלך");
    }

    #[test]
    fn advanced_search_matches_gershayim_tokens_end_to_end() {
        // HebrewTokenizer שומר גרשיים וגרש פנימי בטוקן (ז"ל, רמב"ם — טרם
        // אחד); split_query_words חייב לפצל את השאילתה באותה צורה — אחרת
        // ז"ל לעולם לא יימצא. כולל נורמליזציה של ״ עברי משני הצדדים
        // וקיפול זוג-הגרשים (רמב''ם) בשאילתה.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "הרב פלוני ז\"ל אמר", "/books/a.txt");
        add(&mut engine, 2, "כתב הרמב\u{05F4}ם בהלכות", "/books/b.txt");
        add(&mut engine, 3, "דברי תוס' שם", "/books/c.txt");
        add(&mut engine, 4, "מסמך עם רמב ם כשתי מילים", "/books/d.txt");
        add(&mut engine, 5, "אמר ג'ורג' לד'אש", "/books/e.txt");
        engine.commit().unwrap();

        let advanced_ids = |engine: &mut SearchEngine, query: &str| {
            ids(search_advanced_default(
                &engine,
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap())
        };

        assert_eq!(advanced_ids(&mut engine, "ז\"ל"), vec![1]);
        assert_eq!(
            advanced_ids(&mut engine, "ז\u{05F4}ל"),
            vec![1],
            "גרשיים עבריים בשאילתה"
        );
        assert_eq!(
            advanced_ids(&mut engine, "הרמב\"ם"),
            vec![2],
            "״ עברי באינדקס, \" בשאילתה"
        );
        assert_eq!(
            advanced_ids(&mut engine, "הרמב''ם"),
            vec![2],
            "זוג גרשים בשאילתה (מוסכמת קבצים ישנים)"
        );
        assert_eq!(advanced_ids(&mut engine, "תוס'"), vec![3], "גרש סופי");
        assert_eq!(
            advanced_ids(&mut engine, "ג'ורג'"),
            vec![5],
            "גרש פנימי + סופי"
        );
        assert!(
            !advanced_ids(&mut engine, "רמב\"ם").contains(&4),
            "רמב ם כשתי מילים אינו צירוף מקרי של רמב\"ם"
        );
    }

    #[test]
    fn exact_search_gershayim_token_is_a_single_term() {
        // כל צורות הדפוס של השאילתה מתלכדות לטרם `רמב"ם` אחד ומוצאות
        // מסמך שנדפס ב-״; צירוף מקרי `רמב ם` (שתי מילים) לא נתפס עוד —
        // זו בדיוק מטרת השינוי. המחיר המקובל (D1): שאילתה נטולת-גרשיים
        // לא מוצאת את המהדורה המנוקדת-בגרשיים בחיפוש מדויק.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "דברי רמב\u{05F4}ם בהלכות", "/books/a.txt");
        add(&mut engine, 2, "צירוף רמב ם מקרי", "/books/b.txt");
        engine.commit().unwrap();

        let exact_ids = |engine: &SearchEngine, query: &str| {
            ids(engine
                .search_exact(
                    query.to_string(),
                    vec![],
                    100,
                    0,
                    ResultsOrder::Catalogue,
                    false,
                    false,
                    None,
                )
                .unwrap())
        };

        for query in ["רמב\"ם", "רמב\u{05F4}ם", "רמב''ם"] {
            assert_eq!(exact_ids(&engine, query), vec![1], "query {query:?}");
        }
        // המחיר ההיסטורי של אופציה A (מדויק רגיש-גרשיים) בוטל: האינדקס
        // מטמיע לכל מילת-גרשיים גם טוקן-תאום נקי, כך ששאילתה נטולת-גרשיים
        // מוצאת את המהדורה המנוקדת-בגרשיים גם בחיפוש מדויק.
        assert_eq!(
            exact_ids(&engine, "רמבם"),
            vec![1],
            "הטוקן-התאום נטול-הגרשיים"
        );
        // ביטוי רב-מילים עם טוקן-גרש: PhraseQuery על הטרמים החדשים.
        add(&mut engine, 3, "דברי תוס' ד\"ה אמר שם", "/books/c.txt");
        engine.commit().unwrap();
        assert_eq!(exact_ids(&engine, "תוס' ד\"ה"), vec![3]);
    }

    #[test]
    fn fuzzy_bridges_gershayim_and_clean_editions() {
        // הגישור המקורב: `"` = עריכת codepoint אחת, ו-Fix 2 מזריק את
        // הצורה הנקייה גם במרחק 0.
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "דברי רמב\"ם בהלכות", "/books/a.txt");
        add(&mut engine, 2, "דברי רמבם בהלכות", "/books/b.txt");
        engine.commit().unwrap();

        let fuzzy_ids = |engine: &SearchEngine, query: &str, d: u8| {
            ids(engine
                .search_fuzzy(
                    query.to_string(),
                    vec!["/root".to_string()],
                    100,
                    0,
                    d,
                    ResultsOrder::Relevance,
                    false,
                    false,
                    None,
                )
                .unwrap())
        };

        // במרחק 1 — בשני הכיוונים.
        let got = fuzzy_ids(&engine, "רמב\"ם", 1);
        assert!(got.contains(&1) && got.contains(&2), "got {got:?}");
        let got = fuzzy_ids(&engine, "רמבם", 1);
        assert!(got.contains(&1) && got.contains(&2), "got {got:?}");
        // במרחק 0 — הודות לווריאנט הנקי (Fix 2).
        let got = fuzzy_ids(&engine, "רמב\"ם", 0);
        assert!(
            got.contains(&1) && got.contains(&2),
            "הווריאנט הנקי מגשר גם במרחק 0, got {got:?}"
        );
        // תקציב העריכה לא נבלע ע"י הגרשיים: רמכם→רמבם עריכה אחת — ומאז
        // שהאינדקס מטמיע טוקן-תאום נקי, הטרם רמבם קיים גם במסמך הגרשיים,
        // כך ששני המסמכים נתפסים כבר במרחק 1.
        let got = fuzzy_ids(&engine, "רמכם", 1);
        assert!(
            got.contains(&1) && got.contains(&2),
            "got {got:?}: רמכם→רמבם עריכה אחת, בשני המסמכים"
        );
        let got = fuzzy_ids(&engine, "רמכם", 2);
        assert!(got.contains(&1) && got.contains(&2), "got {got:?}");
    }

    #[test]
    fn lexical_fuzzy_expands_quote_bearing_token() {
        // Fix 1 מקצה-לקצה: מפתח ה-lookup של טוקן-גרשיים פוגע ב-lexical.db
        // (אחרי מחיקת `"` ASCII), וההרחבה הלקסיקלית מזריקה קרובים שמרחק
        // העריכה לבדו לעולם לא היה תופס.
        let (mut engine, dir) = make_engine();
        assert!(engine.set_magic_dictionary_path(make_lexical_db(&dir)));
        add(&mut engine, 1, "דברי אדמורים רבים כאן", "/books/a.txt");
        add(&mut engine, 2, "דברי אדמו\"ר אחד כאן", "/books/b.txt");
        engine.commit().unwrap();

        let got = ids(engine
            .search_fuzzy(
                "אדמו\"ר".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                1,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap());
        assert!(
            got.contains(&1),
            "אדמורים רחוק 3 עריכות — מושג רק דרך ההרחבה הלקסיקלית, got {got:?}"
        );
        assert!(got.contains(&2), "הטרם המדויק עצמו, got {got:?}");

        // הפער השיורי ההיסטורי (§5.2) נסגר: המילון פולט צורות נקיות בלבד,
        // אבל האינדקס מטמיע כעת טוקן-תאום נקי (אדמור) לצד אדמו"ר — כך
        // שהצורה הנקייה שהמילון מחזיר פוגעת בו ישירות.
        let got = ids(engine
            .search_fuzzy(
                "אדמורים".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                1,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap());
        assert!(got.contains(&1), "got {got:?}");
        assert!(
            got.contains(&2),
            "הטוקן-התאום סוגר את הפער השיורי, got {got:?}"
        );
    }

    #[test]
    fn advanced_typo_and_prefix_flags_work_on_gershayim_tokens() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "דברי רמבם בהלכות", "/books/a.txt");
        add(&mut engine, 2, "כתב הרמב\"ם על כך", "/books/b.txt");
        engine.commit().unwrap();

        let search = |engine: &mut SearchEngine, query: &str, opt: &str| {
            let mut word_opts = HashMap::new();
            word_opts.insert(opt.to_string(), true);
            let mut options = HashMap::new();
            options.insert(format!("{query}_0"), word_opts);
            ids(search_advanced_default(
                &engine,
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                options,
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap())
        };

        // דגל typo: וריאנט-המחיקה של הגרפמה `"` מגשר למהדורות נקיות.
        let got = search(&mut engine, "רמב\"ם", hebrew_query::OPT_TYPO);
        assert!(got.contains(&1), "מחיקת `\"` מייצרת את רמבם, got {got:?}");
        // קידומות דקדוקיות סביב שורש עם `"` literal.
        let got = search(&mut engine, "רמב\"ם", "קידומות דקדוקיות");
        assert!(got.contains(&2), "ה־רמב\"ם עם קידומת, got {got:?}");
    }

    #[test]
    fn advanced_aramaic_option_matches_prefixes_and_final_swaps() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "מלכא קדישא", "/books/a.txt");
        add(&mut engine, 2, "אמרו דמלכא הוא", "/books/b.txt");
        add(&mut engine, 3, "כדמלכה בשעתו", "/books/c.txt");
        add(&mut engine, 4, "מדמלכא נפק", "/books/d.txt");
        add(&mut engine, 5, "אדמלכה קאי", "/books/e.txt");
        add(&mut engine, 6, "חכמין אמרין", "/books/f.txt");
        add(&mut engine, 7, "ספרא אחרינא", "/books/g.txt");
        engine.commit().unwrap();

        let search = |engine: &mut SearchEngine, query: &str, opts: &[&str]| {
            let mut options = HashMap::new();
            if !opts.is_empty() {
                let word_opts: HashMap<String, bool> =
                    opts.iter().map(|o| (o.to_string(), true)).collect();
                options.insert(format!("{query}_0"), word_opts);
            }
            ids(search_advanced_default(
                &engine,
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                options,
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap())
        };

        // שתי האפשרויות יחד — ההתנהגות ההיסטורית: שקילות סופית ה↔א +
        // הקידומות ד/כד/מד/אד על שני הווריאנטים.
        let got = search(&mut engine, "מלכה", &["קידומות ארמיות", "סיומות ארמיות"]);
        for id in [1, 2, 3, 4, 5] {
            assert!(got.contains(&id), "ארמית החמיצה את מסמך {id}, got {got:?}");
        }
        assert!(!got.contains(&7), "ארמית רחבה מדי, got {got:?}");

        // סיומות בלבד: השקילות עובדת (מלכא, id 1) אבל אין קידומות (לא 2-5).
        let got = search(&mut engine, "מלכה", &["סיומות ארמיות"]);
        assert!(
            got.contains(&1),
            "סיומות ארמיות החמיצו את מלכא, got {got:?}"
        );
        for id in [2, 3, 4, 5] {
            assert!(
                !got.contains(&id),
                "סיומות בלבד לא אמורות לתת קידומות (מסמך {id}), got {got:?}"
            );
        }

        // קידומות בלבד: דמלכה עם קידומת נתפס דרך וריאנט? לא — אין שקילות
        // סופית, אז רק צורות של "מלכה" עם קידומת; המסמכים כאן נושאים מלכא
        // חוץ מ-3 ו-5 (כדמלכה, אדמלכה).
        let got = search(&mut engine, "מלכה", &["קידומות ארמיות"]);
        for id in [3, 5] {
            assert!(
                got.contains(&id),
                "קידומות ארמיות החמיצו את מסמך {id}, got {got:?}"
            );
        }
        for id in [1, 2, 4] {
            assert!(
                !got.contains(&id),
                "קידומות בלבד לא אמורות לתת שקילות סופית (מסמך {id}), got {got:?}"
            );
        }

        // ם↔ן: חכמים מוצא חכמין דרך סיומות ארמיות.
        let got = search(&mut engine, "חכמים", &["סיומות ארמיות"]);
        assert!(got.contains(&6), "ם↔ן לא עבד, got {got:?}");

        // בלי האפשרויות — אין שקילות ארמית.
        let got = search(&mut engine, "מלכה", &[]);
        assert!(got.is_empty(), "בלי ארמית לא אמור להימצא דבר, got {got:?}");
    }

    #[test]
    fn quote_free_indexing_and_ignore_quotes_option() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "כתב רמב\"ם על כך", "/books/a.txt");
        add(&mut engine, 2, "דברי רמבם בהלכות", "/books/b.txt");
        engine.commit().unwrap();

        let search = |engine: &mut SearchEngine, query: &str, opt: Option<&str>| {
            let mut options = HashMap::new();
            if let Some(opt) = opt {
                options.insert(
                    format!("{query}_0"),
                    HashMap::from([(opt.to_string(), true)]),
                );
            }
            ids(search_advanced_default(
                &engine,
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                options,
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap())
        };

        // בלי שום אפשרות: הצורה הנקייה מוצאת גם את המקור עם הגרשיים —
        // הטוקן-התאום שהאינדקס מטמיע.
        let got = search(&mut engine, "רמבם", None);
        assert!(got.contains(&1) && got.contains(&2), "got {got:?}");

        // רמב"ם בלי האפשרות: התנהגות היסטורית — רק הצורה עם הגרשיים.
        let got = search(&mut engine, "רמב\"ם", None);
        assert!(got.contains(&1) && !got.contains(&2), "got {got:?}");

        // עם "התעלם מגרשיים": שתי הצורות.
        let got = search(&mut engine, "רמב\"ם", Some("התעלם מגרשיים"));
        assert!(got.contains(&1) && got.contains(&2), "got {got:?}");

        // ההדגשה מכסה את הצורה המקורית עם הגרשיים (התאום יורש offsets).
        let results = search_advanced_default(
            &engine,
            "רמבם".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap();
        // הטקסט עובר escape של HTML — הגרשיים מופיעות כ-&quot;.
        let doc1 = results.iter().find(|r| r.id == 1).unwrap();
        assert!(
            doc1.text.contains("<font color=red>רמב&quot;ם</font>"),
            "highlight: {}",
            doc1.text
        );
    }

    #[test]
    fn advanced_translation_option_expands_from_dictionary() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "איתא בגמרא", "/books/a.txt");
        add(&mut engine, 2, "יש דברים בגו", "/books/b.txt");
        engine.commit().unwrap();

        let mut dict_file = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write as _;
        dict_file
            .write_all(r#"{ "מילון פשיטא": [ { "אִיתָא": "יש" } ] }"#.as_bytes())
            .unwrap();
        assert!(
            engine.set_translation_dictionary_path(dict_file.path().to_string_lossy().into_owned())
        );
        assert!(engine.has_translation_dictionary());

        let search = |engine: &mut SearchEngine, query: &str, opt: Option<&str>| {
            let mut options = HashMap::new();
            if let Some(opt) = opt {
                options.insert(
                    format!("{query}_0"),
                    HashMap::from([(opt.to_string(), true)]),
                );
            }
            ids(search_advanced_default(
                &engine,
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                options,
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap())
        };

        // עברי→ארמי: "יש" עם תרגום ארמי מוצא גם את "איתא".
        let got = search(&mut engine, "יש", Some("תרגום ארמי"));
        assert!(got.contains(&1) && got.contains(&2), "got {got:?}");
        // בלי האפשרות — אין הרחבה.
        let got = search(&mut engine, "יש", None);
        assert!(!got.contains(&1) && got.contains(&2), "got {got:?}");
        // ארמי→עברי.
        let got = search(&mut engine, "איתא", Some("תרגום ארמי"));
        assert!(got.contains(&1) && got.contains(&2), "got {got:?}");
    }

    #[test]
    fn advanced_acronym_option_expands_bidirectionally() {
        let (mut engine, _dir) = make_engine();
        // id 1 — הר"ת עצמו (מאונדקס גם כ-"רמבם" דרך הטוקן-התאום).
        add(&mut engine, 1, "אמר רמב\"ם בהלכות", "/books/a.txt");
        // id 2 — הפענוח המלא ככתוב.
        add(
            &mut engine,
            2,
            "כתב רבי משה בן מיימון בספרו",
            "/books/b.txt",
        );
        // id 3 — לא קשור.
        add(&mut engine, 3, "דבר אחר לגמרי", "/books/c.txt");
        engine.commit().unwrap();

        let mut dict_file = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write as _;
        dict_file
            .write_all(r#"{ "רמב\"ם": ["רבי משה בן מיימון"] }"#.as_bytes())
            .unwrap();
        assert!(
            engine.set_acronyms_dictionary_path(dict_file.path().to_string_lossy().into_owned())
        );
        assert!(engine.has_acronyms_dictionary());

        // האפשרות דלוקה על המילה במיקום `word_index`.
        let search = |engine: &SearchEngine, query: &str, opt_on_word: Option<(&str, usize)>| {
            let mut options = HashMap::new();
            if let Some((word, i)) = opt_on_word {
                options.insert(
                    format!("{word}_{i}"),
                    HashMap::from([("ראשי תיבות".to_string(), true)]),
                );
            }
            ids(search_advanced_default(
                engine,
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                options,
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap())
        };

        // כיוון א' (ר"ת→פענוח): "רמב\"ם" עם האפשרות מוצא גם את הפענוח המלא.
        let got = search(&engine, "רמב\"ם", Some(("רמב\"ם", 0)));
        assert!(got.contains(&1) && got.contains(&2), "forward: {got:?}");
        assert!(!got.contains(&3), "forward must not over-match: {got:?}");
        // בלי האפשרות — רק ההתאמה הישירה.
        let got = search(&engine, "רמב\"ם", None);
        assert!(got.contains(&1) && !got.contains(&2), "no-opt: {got:?}");

        // כיוון ב' (פענוח→ר"ת): הביטוי המלא עם האפשרות מוצא גם את הר"ת.
        let got = search(&engine, "רבי משה בן מיימון", Some(("רבי", 0)));
        assert!(got.contains(&1) && got.contains(&2), "reverse: {got:?}");
        // בלי האפשרות — רק ההתאמה הישירה לביטוי.
        let got = search(&engine, "רבי משה בן מיימון", None);
        assert!(
            !got.contains(&1) && got.contains(&2),
            "reverse no-opt: {got:?}"
        );

        // הדגשה: מסמך שנמצא דרך החלופה נצבע — בשני הכיוונים.
        let texts = |query: &str, word: &str| -> Vec<String> {
            let options = HashMap::from([(
                format!("{word}_0"),
                HashMap::from([("ראשי תיבות".to_string(), true)]),
            )]);
            search_advanced_default(
                &engine,
                query.to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                options,
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.text)
            .collect()
        };
        // ר"ת→פענוח: מילות הפענוח נצבעות במסמך 2.
        let got = texts("רמב\"ם", "רמב\"ם");
        assert!(
            got.iter().any(|t| t.contains("<font color=red>רבי</font>")
                && t.contains("<font color=red>מיימון</font>")),
            "forward highlight missing: {got:?}"
        );
        // וגם הר\"ת עצמו נצבע במסמך 1.
        assert!(
            got.iter().any(|t| t.contains("<font color=red>רמב")),
            "forward self-highlight missing: {got:?}"
        );
        // פענוח→ר"ת: הר"ת נצבע במסמך 1.
        let got = texts("רבי משה בן מיימון", "רבי");
        assert!(
            got.iter().any(|t| t.contains("<font color=red>רמב")),
            "reverse highlight missing: {got:?}"
        );
    }

    #[test]
    fn test_search_fuzzy_high_level() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום", "/books/a.txt");
        add(&mut engine, 2, "שלם", "/books/b.txt");
        add(&mut engine, 3, "ביי", "/books/c.txt");
        engine.commit().unwrap();

        let texts: Vec<String> = engine
            .search_fuzzy(
                "שלום".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                1,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.text)
            .collect();
        // Fuzzy matching is automaton-based, so highlighting must come from the
        // materialized highlight query: both the exact match and the variant one
        // edit away are wrapped, not just the literal word the user typed.
        assert!(texts.contains(&"<font color=red>שלום</font>".to_string()));
        assert!(texts.contains(&"<font color=red>שלם</font>".to_string()));
        assert!(!texts.iter().any(|t| t.contains("ביי")));
    }

    #[test]
    fn test_search_fuzzy_invalid_distance_errors() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום", "/books/a.txt");
        engine.commit().unwrap();

        // Tantivy supports edit distances 0–2; anything above must surface as
        // an error from every fuzzy entry point, never a panic.
        let result = engine.search_fuzzy(
            "שלום".to_string(),
            vec!["/root".to_string()],
            10,
            0,
            3,
            ResultsOrder::Relevance,
            false,
            false,
            None,
        );
        assert!(result.is_err(), "distance > 2 should error, not panic");

        let count = engine.count_fuzzy(
            "שלום".to_string(),
            vec!["/root".to_string()],
            3,
            false,
            false,
        );
        assert!(count.is_err(), "distance > 2 should error, not panic");
    }

    #[test]
    fn test_search_fuzzy_highlights_near_match_in_context() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "אמר שלם לחברו", "/books/a.txt");
        engine.commit().unwrap();

        let results = engine
            .search_fuzzy(
                "שלום".to_string(),
                vec!["/root".to_string()],
                10,
                0,
                1,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].text, "אמר <font color=red>שלם</font> לחברו",
            "the fuzzy-matched variant must be highlighted inside the snippet"
        );
    }

    #[test]
    fn test_search_fuzzy_empty_query_returns_no_results() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום", "/books/a.txt");
        engine.commit().unwrap();

        // Mirror exact mode: empty/punctuation-only fuzzy queries match
        // nothing instead of returning every document in the facets.
        for query in ["", "?!"] {
            let results = engine
                .search_fuzzy(
                    query.to_string(),
                    vec!["/root".to_string()],
                    100,
                    0,
                    1,
                    ResultsOrder::Relevance,
                    false,
                    false,
                    None,
                )
                .unwrap();
            assert!(results.is_empty(), "query {query:?} should match nothing");
        }
    }

    #[test]
    fn test_high_level_counts() {
        let (mut engine, _dir) = make_engine();
        add(&mut engine, 1, "שלום עולם", "/books/a.txt");
        add(&mut engine, 2, "שלום רב", "/books/a.txt");
        add(&mut engine, 3, "ביי", "/books/b.txt");
        engine.commit().unwrap();

        assert_eq!(
            engine
                .count_exact("שלום".to_string(), vec!["/root".to_string()], false, false)
                .unwrap(),
            2
        );

        let by_book = engine
            .count_by_book_exact("שלום".to_string(), vec!["/root".to_string()], false, false)
            .unwrap();
        assert_eq!(by_book.get("/books/a.txt").copied(), Some(2));

        let page = engine
            .search_and_count_exact(
                "שלום".to_string(),
                vec!["/root".to_string()],
                1,
                0,
                ResultsOrder::Relevance,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(page.total_count, 2);
        assert_eq!(page.results.len(), 1);
    }

    // ── חיפוש מנוקד (textVocalized) ─────────────────────────────────────

    /// ספר קטן: שורה מנוקדת, שורה נטולת ניקוד עם אותן מילים, שורה עם
    /// ניקוד+טעמים, ושורה מנוקדת עם קידומת.
    fn make_vocalized_engine() -> (SearchEngine, TempDir) {
        let (mut engine, dir) = make_engine();
        let text = "<h1>בראשית</h1>\n\
                    בְּרֵאשִׁית בָּרָא אֱלֹהִים\n\
                    ובראשית ברא אלהים בלי ניקוד\n\
                    וַיֹּ\u{05A3}אמֶר אֱלֹהִים יְהִי אוֹר\n\
                    וּבָרָא עוֹלָם";
        engine
            .add_text_book(
                "בראשית".to_string(),
                "/tanakh".to_string(),
                "/books/b.txt".to_string(),
                1,
                DEFAULT_GENERATION_ORDER,
                text.to_string(),
                None,
            )
            .unwrap();
        engine.commit().unwrap();
        (engine, dir)
    }

    #[test]
    fn vocalized_exact_requires_typed_marks_frees_untyped() {
        let (engine, _dir) = make_vocalized_engine();
        // קמץ שהוקלד חייב; הדגש שלא הוקלד חופשי — בָרָא מוצא את בָּרָא.
        let hits = engine
            .search_exact(
                "בָרָא".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                true,
                false,
                None,
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        // התוצאה מציגה את העותק המנוקד השמור.
        assert!(hits[0].text.contains("בָּרָא"), "text: {}", hits[0].text);

        // תנועה שגויה במקום קמץ — נפסל.
        let miss = engine
            .search_exact(
                "בֵרָא".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                true,
                false,
                None,
            )
            .unwrap();
        assert!(miss.is_empty());
    }

    #[test]
    fn vocalized_flag_off_keeps_plain_behaviour() {
        let (engine, _dir) = make_vocalized_engine();
        // בלי דגלים: הניקוד מנורמל החוצה והחיפוש מוצא את שתי השורות.
        let plain = engine
            .search_exact(
                "ברא".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(plain.len(), 2);
        // עם דגל ניקוד ושאילתה לא מנוקדת: כל סימן חופשי, אך רק שורות
        // מנוקדות קיימות בשדה — השורה הנקייה לא נמצאת.
        let voc = engine
            .search_exact(
                "ברא".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                true,
                false,
                None,
            )
            .unwrap();
        assert_eq!(voc.len(), 1);
    }

    #[test]
    fn vocalized_taamim_flag_splits_classes() {
        let (engine, _dir) = make_vocalized_engine();
        // שאילתה מנוקדת בלי טעמים מוצאת טקסט עם טעם (הטעם חופשי).
        let nikud_only = engine
            .search_exact(
                "וַיֹּאמֶר".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                true,
                false,
                None,
            )
            .unwrap();
        assert_eq!(nikud_only.len(), 1);
        // שאילתה עם הטעם שהוקלד ושני הדגלים — עדיין נמצא (הטעם קיים בטקסט).
        let with_taam = engine
            .search_exact(
                "וַיֹּ\u{05A3}אמֶר".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                true,
                true,
                None,
            )
            .unwrap();
        assert_eq!(with_taam.len(), 1);
        // טעם שגוי (זקף-קטן במקום מונח) עם דגל טעמים — נפסל.
        let wrong_taam = engine
            .search_exact(
                "וַיֹּ\u{0594}אמֶר".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                true,
                true,
                None,
            )
            .unwrap();
        assert!(wrong_taam.is_empty());
    }

    #[test]
    fn vocalized_exact_phrase_matches_and_counts() {
        let (engine, _dir) = make_vocalized_engine();
        let page = engine
            .search_and_count_exact(
                "בָּרָא אֱלֹהִים".to_string(),
                vec![],
                10,
                0,
                ResultsOrder::Catalogue,
                true,
                false,
                None,
            )
            .unwrap();
        assert_eq!(page.total_count, 1);
        assert_eq!(page.results.len(), 1);

        let by_book = engine
            .count_by_book_exact("בָּרָא".to_string(), vec![], true, false)
            .unwrap();
        assert_eq!(by_book.get("/books/b.txt"), Some(&1));
    }

    #[test]
    fn vocalized_advanced_prefix_option_matches_prefixed_word() {
        let (engine, _dir) = make_vocalized_engine();
        let options = HashMap::from([(
            "בָרָא_0".to_string(),
            HashMap::from([("קידומות".to_string(), true)]),
        )]);
        let hits = search_advanced_default(
            &engine,
            "בָרָא".to_string(),
            vec![],
            10,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            options,
            ResultsOrder::Catalogue,
            true,
            false,
            SearchScope::WordDistance,
        )
        .unwrap();
        // גם בָּרָא וגם וּבָרָא (הקידומת המנוקדת בתוך חלון ה-prefix).
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn per_word_nikud_option_routes_to_vocalized_field() {
        let (engine, _dir) = make_vocalized_engine();
        let search = |query: &str, options: HashMap<String, HashMap<String, bool>>| {
            search_advanced_default(
                &engine,
                query.to_string(),
                vec![],
                10,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                options,
                ResultsOrder::Catalogue,
                // הדגלים הגלובליים כבויים — הבקשה מגיעה מהאפשרות הפר-מילה.
                false,
                false,
                SearchScope::WordDistance,
            )
            .unwrap()
        };
        // בלי האפשרות: מסלול רגיל, הניקוד נבלע בנרמול — מוצא את שתי השורות
        // (המנוקדת והלא-מנוקדת).
        assert_eq!(search("בָרָא", HashMap::new()).len(), 2);
        // עם "ניקוד" על המילה: רץ על השדה המנוקד ודורש את הקמץ שהוקלד —
        // רק השורה המנוקדת.
        let options = HashMap::from([(
            "בָרָא_0".to_string(),
            HashMap::from([(hebrew_query::OPT_MATCH_NIKUD.to_string(), true)]),
        )]);
        let hits = search("בָרָא", options);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("בָּרָא"), "text: {}", hits[0].text);
        // תנועה שגויה עם האפשרות — נפסל.
        let options = HashMap::from([(
            "בֻרָא_0".to_string(),
            HashMap::from([(hebrew_query::OPT_MATCH_NIKUD.to_string(), true)]),
        )]);
        assert!(search("בֻרָא", options).is_empty());
    }

    #[test]
    fn per_word_nikud_option_leaves_other_words_free() {
        let (engine, _dir) = make_vocalized_engine();
        // "ניקוד" מסומן רק על המילה הראשונה; השנייה מוקלדת בניקוד "שגוי"
        // בכוונה — הסימנים שלה חופשיים ולכן ההתאמה שורדת.
        let options = HashMap::from([(
            "בָּרָא_0".to_string(),
            HashMap::from([(hebrew_query::OPT_MATCH_NIKUD.to_string(), true)]),
        )]);
        let hits = search_advanced_default(
            &engine,
            "בָּרָא אֱלֹהִֻים".to_string(),
            vec![],
            10,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            options,
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::WordDistance,
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("בְּרֵאשִׁית"), "text: {}", hits[0].text);
    }

    #[test]
    fn vocalized_fuzzy_finds_edit_distance_variants() {
        let (engine, _dir) = make_vocalized_engine();
        // ברח במרחק עריכה 1 מ-ברא; הווריאנט חופשי-סימנים מוצא את בָּרָא.
        let hits = engine
            .search_fuzzy(
                "בָּרַח".to_string(),
                vec![],
                10,
                0,
                1,
                ResultsOrder::Catalogue,
                true,
                false,
                None,
            )
            .unwrap();
        assert!(!hits.is_empty());
        // מרחק 0: רק הצורה המדויקת (עם הסימנים שהוקלדו) — ברח לא קיים.
        let exact_only = engine
            .search_fuzzy(
                "בָּרַח".to_string(),
                vec![],
                10,
                0,
                0,
                ResultsOrder::Catalogue,
                true,
                false,
                None,
            )
            .unwrap();
        assert!(exact_only.is_empty());
    }

    // ── SearchScope: "באותה פסקה" / "תחת אותה כותרת" ─────────────────────

    /// ספר עם שתי כותרות משנה: תחת "סימן א" המילים "מתכוין" ו"מתעסק"
    /// מפוזרות על שורות שונות; תחת "סימן ב" שורה אחת מכילה את שתיהן —
    /// בסדר הפוך לסדר השאילתה.
    fn scope_engine() -> (SearchEngine, TempDir, u64) {
        let (mut engine, dir) = make_engine();
        let text = "<h1>ספר הבדיקה</h1>\n\
                    <h2>סימן א</h2>\n\
                    דין אינו מתכוין בשבת\n\
                    ודין מתעסק בחלבים ועריות\n\
                    <h2>סימן ב</h2>\n\
                    כאן נדון רק במלאכת שבת\n\
                    מתעסק וגם אינו מתכוין באותה שורה";
        engine
            .add_text_book(
                "ספר הבדיקה".to_string(),
                "/root".to_string(),
                "/books/scope.txt".to_string(),
                7,
                0,
                text.to_string(),
                None,
            )
            .unwrap();
        engine.commit().unwrap();
        // ids: ((catalogue_order+1) << 32) + ordinal + 1
        let id_base = (7u64 + 1) << 32;
        (engine, dir, id_base)
    }

    fn scope_search(engine: &SearchEngine, query: &str, scope: SearchScope) -> Vec<u64> {
        search_advanced_default(
            &engine,
            query.to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            scope,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect()
    }

    #[test]
    fn same_paragraph_scope_matches_unordered_within_line() {
        let (engine, _dir, id_base) = scope_engine();
        // בסדר השאילתה אין התאמה בשום שורה — מסלול המרחק (סדר + צמידות)
        // לא מוצא דבר.
        assert_eq!(
            scope_search(&engine, "מתכוין מתעסק", SearchScope::WordDistance),
            Vec::<u64>::new()
        );
        // באותה פסקה: רק השורה שמכילה את שתי המילים, למרות הסדר ההפוך.
        assert_eq!(
            scope_search(&engine, "מתכוין מתעסק", SearchScope::SameParagraph),
            vec![id_base + 7]
        );
    }

    #[test]
    fn same_section_scope_matches_across_lines_under_one_heading() {
        let (engine, _dir, id_base) = scope_engine();
        // "מתכוין" ו"מתעסק" בשורות שונות תחת "סימן א", ובאותה שורה תחת
        // "סימן ב" — חוזרות כל השורות שנושאות מילה מהשאילתה בתוך סעיף
        // שמכיל את כל המילים.
        assert_eq!(
            scope_search(&engine, "מתכוין מתעסק", SearchScope::SameSection),
            vec![id_base + 3, id_base + 4, id_base + 7]
        );
        // "בשבת" (סימן א) ו"נדון" (סימן ב) — אף סעיף לא מכיל את שתיהן.
        assert_eq!(
            scope_search(&engine, "בשבת נדון", SearchScope::SameSection),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn same_section_negative_scope_excludes_whole_section() {
        let (engine, _dir, id_base) = scope_engine();
        // "מתכוין" מופיע בסימן א (שורה 3) ובסימן ב (שורה 7). צירוף השלילה
        // "מתעסק בחלבים" נמצא בשורה *אחרת* של סימן א (שורה 4) — שלילה בטווח
        // "תחת אותה כותרת" חייבת לפסול את כל הסעיף, כולל שורות שאין בהן
        // אף מילת שלילה, ולהותיר רק את סימן ב.
        let ids: Vec<u64> = engine
            .search_advanced(
                "מתכוין".to_string(),
                "מתעסק בחלבים".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::SameSection,
                SearchScope::SameSection,
                None,
                None,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec![id_base + 7]);

        // שלילה שאינה חותכת אף סעיף אינה משנה דבר.
        let ids: Vec<u64> = engine
            .search_advanced(
                "מתכוין".to_string(),
                "בחלבים נדון".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::SameSection,
                SearchScope::SameSection,
                None,
                None,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec![id_base + 3, id_base + 7]);
    }

    #[test]
    fn same_section_scope_count_matches_search() {
        let (engine, _dir, _id_base) = scope_engine();
        let count = count_advanced_default(
            &engine,
            "מתכוין מתעסק".to_string(),
            vec!["/root".to_string()],
            0,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            false,
            false,
            SearchScope::SameSection,
        )
        .unwrap();
        assert_eq!(count, 3);

        let by_book = count_by_book_advanced_default(
            &engine,
            "מתכוין מתעסק".to_string(),
            vec!["/root".to_string()],
            0,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            false,
            false,
            SearchScope::SameSection,
        )
        .unwrap();
        assert_eq!(by_book.get("/books/scope.txt"), Some(&3));
    }

    #[test]
    fn same_paragraph_scope_snippet_highlights_every_word() {
        let (engine, _dir, _id_base) = scope_engine();
        let results = search_advanced_default(
            &engine,
            "מתכוין מתעסק".to_string(),
            vec!["/root".to_string()],
            100,
            0,
            0,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            ResultsOrder::Catalogue,
            false,
            false,
            SearchScope::SameParagraph,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        let text = &results[0].text;
        // שתי המילים מודגשות, בלי מסנן סדר/מרחק.
        assert!(
            text.contains("<font color=red>מתעסק</font>"),
            "snippet: {text}"
        );
        assert!(
            text.contains("<font color=red>מתכוין</font>"),
            "snippet: {text}"
        );
    }

    fn word_match_results(
        engine: &SearchEngine,
        query: &str,
        scope: SearchScope,
        mode: Option<WordMatchMode>,
        count: Option<u32>,
    ) -> Vec<SearchResult> {
        let negative_scope = same_search_scope(&scope);
        engine
            .search_advanced(
                query.to_string(),
                String::new(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                ResultsOrder::Catalogue,
                false,
                false,
                scope,
                negative_scope,
                None,
                mode,
                count,
            )
            .unwrap()
    }

    fn word_match_search(
        engine: &SearchEngine,
        query: &str,
        scope: SearchScope,
        mode: Option<WordMatchMode>,
        count: Option<u32>,
    ) -> Vec<u64> {
        word_match_results(engine, query, scope, mode, count)
            .into_iter()
            .map(|r| r.id)
            .collect()
    }

    #[test]
    fn any_word_matches_lines_with_a_single_query_word() {
        let (engine, _dir, id_base) = scope_engine();
        // ברירת המחדל (כל המילים, מסלול המרחק) — אין שורה עם הצירוף.
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין בחלבים",
                SearchScope::WordDistance,
                None,
                None
            ),
            Vec::<u64>::new()
        );
        // "מילה אחת מספיקה": כל שורה שנושאת אחת מהמילים, בכל scope.
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין בחלבים",
                SearchScope::WordDistance,
                Some(WordMatchMode::AnyWord),
                None
            ),
            vec![id_base + 3, id_base + 4, id_base + 7]
        );
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין בחלבים",
                SearchScope::SameParagraph,
                Some(WordMatchMode::AnyWord),
                None
            ),
            vec![id_base + 3, id_base + 4, id_base + 7]
        );
    }

    #[test]
    fn any_word_snippet_highlights_the_present_word() {
        let (engine, _dir, id_base) = scope_engine();
        let results = word_match_results(
            &engine,
            "מתכוין בחלבים",
            SearchScope::WordDistance,
            Some(WordMatchMode::AnyWord),
            None,
        );
        // שורה שנושאת רק מילה אחת — היא נצבעת, בלי מסנן סדר/מרחק.
        let first = results.iter().find(|r| r.id == id_base + 3).unwrap();
        assert!(
            first.text.contains("<font color=red>מתכוין</font>"),
            "snippet: {}",
            first.text
        );
    }

    #[test]
    fn most_words_requires_a_majority() {
        let (engine, _dir, id_base) = scope_engine();
        // שלוש מילים — רוב = 2. שורה 3 נושאת רק "מתכוין" ונפסלת;
        // שורה 4 נושאת "מתעסק"+"בחלבים"; שורה 7 "מתעסק"+"מתכוין".
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין מתעסק בחלבים",
                SearchScope::WordDistance,
                Some(WordMatchMode::MostWords),
                None
            ),
            vec![id_base + 4, id_base + 7]
        );
    }

    #[test]
    fn at_least_count_is_clamped_and_defaults_to_one() {
        let (engine, _dir, id_base) = scope_engine();
        let at_least = |count: Option<u32>| {
            word_match_search(
                &engine,
                "מתכוין מתעסק בחלבים",
                SearchScope::WordDistance,
                Some(WordMatchMode::AtLeast),
                count,
            )
        };
        // אף שורה לא נושאת את שלוש המילים.
        assert_eq!(at_least(Some(3)), Vec::<u64>::new());
        // מעל מספר המילים — נחתך ל"כולן".
        assert_eq!(at_least(Some(10)), Vec::<u64>::new());
        assert_eq!(at_least(Some(2)), vec![id_base + 4, id_base + 7]);
        // בלי ספירה (או 0) — מילה אחת.
        let any = vec![id_base + 3, id_base + 4, id_base + 7];
        assert_eq!(at_least(Some(1)), any);
        assert_eq!(at_least(None), any);
        assert_eq!(at_least(Some(0)), any);
    }

    #[test]
    fn word_match_same_section_counts_distinct_words_per_section() {
        let (engine, _dir, id_base) = scope_engine();
        // "מתכוין בחלבים נדון", רוב = 2: סימן א נושא מתכוין+בחלבים,
        // סימן ב נושא נדון+מתכוין — שני הסעיפים עוברים, וחוזרות כל
        // השורות שנושאות מילת שאילתה בתוכם.
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין בחלבים נדון",
                SearchScope::SameSection,
                Some(WordMatchMode::MostWords),
                None
            ),
            vec![id_base + 3, id_base + 4, id_base + 6, id_base + 7]
        );
        // כל שלוש המילים — אף סעיף לא מכיל את שלושתן.
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין בחלבים נדון",
                SearchScope::SameSection,
                Some(WordMatchMode::AtLeast),
                Some(3)
            ),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn word_match_all_explicit_matches_default_behavior() {
        let (engine, _dir, id_base) = scope_engine();
        // Some(All) זהה ל-None — מסלול הביטוי הקיים נשאר בתוקף.
        assert_eq!(
            word_match_search(
                &engine,
                "אינו מתכוין",
                SearchScope::WordDistance,
                Some(WordMatchMode::All),
                None
            ),
            vec![id_base + 3, id_base + 7]
        );
        assert_eq!(
            word_match_search(
                &engine,
                "אינו מתכוין",
                SearchScope::WordDistance,
                None,
                None
            ),
            vec![id_base + 3, id_base + 7]
        );
    }

    #[test]
    fn any_word_respects_the_negative_query() {
        let (engine, _dir, id_base) = scope_engine();
        // חיובי בהתאמה חלקית; השלילה נשארת "כל המילים" ופוסלת את שורה 3.
        let ids: Vec<u64> = engine
            .search_advanced(
                "מתכוין בחלבים".to_string(),
                "בשבת".to_string(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
                SearchScope::WordDistance,
                None,
                Some(WordMatchMode::AnyWord),
                None,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec![id_base + 4, id_base + 7]);
    }

    /// שקילות בין מה שהחיפוש מוצא לבין מה שההדגשה בספר הפתוח מסמנת.
    ///
    /// שני הצדדים סופרים "כמה מילים מותר בין מילות השאילתה": החיפוש לפי
    /// עמדות הטוקנים באינדקס, וההדגשה לפי תבנית רגקס על טקסט התצוגה. סטייה
    /// ביניהם מציגה למשתמש מילים מודגשות לצד "אין תוצאות" (או ההפוך).
    #[test]
    fn display_highlight_distance_matches_search_distance() {
        let (mut engine, _dir) = make_engine();
        // כל שורה: פער אחר בין "תדע" ל"זרעך" — כולל מקף, גרשיים וניקוד.
        let lines = [
            "תדע זרעך",
            "תדע אחת זרעך",
            "תדע אחת שתים זרעך",
            "תדע כי־גר יהיה זרעך",
            "תדע רמב\u{05F4}ם זרעך",
            "תֵּדַע יָדֹעַ זַרְעֲךָ",
        ];
        engine
            .add_text_book(
                "ספר הבדיקה".to_string(),
                "/root".to_string(),
                "/books/highlight_parity.txt".to_string(),
                9,
                0,
                lines.join("\n"),
                None,
            )
            .unwrap();
        engine.commit().unwrap();

        for (index, line) in lines.iter().enumerate() {
            let mut found_by_search: Option<u32> = None;
            let mut found_by_highlight: Option<u32> = None;
            for distance in 0..=4u32 {
                if found_by_search.is_none() {
                    let hits = search_advanced_default(
                        &engine,
                        "תדע זרעך".to_string(),
                        vec!["/root".to_string()],
                        100,
                        0,
                        distance,
                        HashMap::new(),
                        HashMap::new(),
                        HashMap::new(),
                        ResultsOrder::Catalogue,
                        false,
                        false,
                        SearchScope::WordDistance,
                    )
                    .unwrap();
                    if hits.iter().any(|hit| hit.segment as usize == index) {
                        found_by_search = Some(distance);
                    }
                }
                if found_by_highlight.is_none() {
                    let highlight = crate::display_highlight::build_display_highlight(
                        "תדע זרעך",
                        distance,
                        &HashMap::new(),
                        &HashMap::new(),
                        &HashMap::new(),
                    )
                    .unwrap();
                    // fancy-regex: בתבנית המשולבת יש lookahead, כמו ב-Dart.
                    let pattern = fancy_regex::Regex::new(&highlight.combined_pattern).unwrap();
                    if pattern.is_match(line).unwrap() {
                        found_by_highlight = Some(distance);
                    }
                }
            }
            assert_eq!(
                found_by_highlight, found_by_search,
                "שורה {line:?}: החיפוש וההדגשה חייבים להתאים באותו מרווח"
            );
        }
    }

    #[test]
    fn word_match_count_matches_search() {
        let (engine, _dir, _id_base) = scope_engine();
        let count = engine
            .count_advanced(
                "מתכוין בחלבים".to_string(),
                String::new(),
                vec!["/root".to_string()],
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                false,
                false,
                SearchScope::WordDistance,
                SearchScope::WordDistance,
                Some(WordMatchMode::AnyWord),
                None,
            )
            .unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn partial_modes_drop_order_even_when_the_threshold_is_all_words() {
        let (engine, _dir, id_base) = scope_engine();
        // שתי מילים: רוב = 2 = כולן — ובכל זאת הסדר אינו נדרש: שורה 7
        // נושאת אותן בסדר הפוך, ומסלול הביטוי (ברירת המחדל) מפספס אותה.
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין מתעסק",
                SearchScope::WordDistance,
                Some(WordMatchMode::MostWords),
                None
            ),
            vec![id_base + 7]
        );
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין מתעסק",
                SearchScope::WordDistance,
                Some(WordMatchMode::AtLeast),
                Some(2)
            ),
            vec![id_base + 7]
        );
    }

    #[test]
    fn duplicate_query_words_count_once_toward_the_threshold() {
        let (engine, _dir, id_base) = scope_engine();
        // "מתכוין מתכוין בחלבים" עם סף 2 — שורה שנושאת רק "מתכוין" אינה
        // עוברת בזכות הכפילות, ואין שורה עם שתי המילים הייחודיות.
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין מתכוין בחלבים",
                SearchScope::WordDistance,
                Some(WordMatchMode::AtLeast),
                Some(2)
            ),
            Vec::<u64>::new()
        );
        // בטווח הסעיף: סימן א מכיל את שתי הייחודיות (בשורות שונות);
        // סימן ב, שבו רק "מתכוין", נפסל למרות הכפילות.
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין מתכוין בחלבים",
                SearchScope::SameSection,
                Some(WordMatchMode::AtLeast),
                Some(2)
            ),
            vec![id_base + 3, id_base + 4]
        );
    }

    #[test]
    fn duplicate_words_with_different_options_still_count_once() {
        let (engine, _dir, _id_base) = scope_engine();
        // אפשרות פר-מילה על המופע הראשון בלבד נותנת לשני המופעים תבניות
        // שונות — ובכל זאת הם מילת-מקור אחת, והסף לא מסופק פעמיים.
        let options: HashMap<String, HashMap<String, bool>> = HashMap::from([(
            "מתכוין_0".to_string(),
            HashMap::from([("קידומות".to_string(), true)]),
        )]);
        let ids: Vec<u64> = engine
            .search_advanced(
                "מתכוין מתכוין בחלבים".to_string(),
                String::new(),
                vec!["/root".to_string()],
                100,
                0,
                0,
                0,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                options,
                HashMap::new(),
                ResultsOrder::Catalogue,
                false,
                false,
                SearchScope::WordDistance,
                SearchScope::WordDistance,
                None,
                Some(WordMatchMode::AtLeast),
                Some(2),
            )
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, Vec::<u64>::new());
    }

    #[test]
    fn any_word_in_section_scope_matches_lines_with_any_word() {
        let (engine, _dir, id_base) = scope_engine();
        // סף 1 בטווח הסעיף שקול ל-OR שטוח (המימוש מדלג על ה-pre-pass).
        assert_eq!(
            word_match_search(
                &engine,
                "מתכוין בחלבים",
                SearchScope::SameSection,
                Some(WordMatchMode::AnyWord),
                None
            ),
            vec![id_base + 3, id_base + 4, id_base + 7]
        );
    }

    // ── Dimension facets (author/era/base) ────────────────────────────────

    fn add_with_facets(
        engine: &mut SearchEngine,
        id: u64,
        text: &str,
        topics: &str,
        file_path: &str,
        extra: &[&str],
    ) {
        engine
            .add_document(
                id,
                "title",
                "ref",
                topics,
                text,
                0,
                false,
                file_path,
                None,
                None,
                Some(extra.iter().map(|s| s.to_string()).collect()),
            )
            .unwrap();
    }

    fn exact_count(engine: &SearchEngine, query: &str, facets: &[&str]) -> u32 {
        engine
            .count_exact(
                query.to_string(),
                facets.iter().map(|s| s.to_string()).collect(),
                false,
                false,
            )
            .unwrap()
    }

    #[test]
    fn dimension_facets_or_within_and_across() {
        let (mut engine, _dir) = make_engine();
        // ספר א: קטגוריה x, ראשונים, רש"י. ספר ב: קטגוריה x, אחרונים.
        // ספר ג: קטגוריה y, ראשונים, וגם ספר-יסוד.
        add_with_facets(
            &mut engine,
            1,
            "דבר המלך",
            "/root/x",
            "a",
            &["/era/ראשונים", "/author/רשי"],
        );
        add_with_facets(
            &mut engine,
            2,
            "דבר המלך",
            "/root/x",
            "b",
            &["/era/אחרונים"],
        );
        add_with_facets(
            &mut engine,
            3,
            "דבר המלך",
            "/root/y",
            "c",
            &["/era/ראשונים", "/base"],
        );
        engine.commit().unwrap();

        // בלי פילטר — הכל.
        assert_eq!(exact_count(&engine, "המלך", &[]), 3);
        // ממד תקופה: OR בתוך הממד.
        assert_eq!(exact_count(&engine, "המלך", &["/era/ראשונים"]), 2);
        assert_eq!(
            exact_count(&engine, "המלך", &["/era/ראשונים", "/era/אחרונים"]),
            3
        );
        // תקופה AND קטגוריה.
        assert_eq!(
            exact_count(&engine, "המלך", &["/era/ראשונים", "/root/x"]),
            1
        );
        // מחבר AND תקופה סותרת — ריק.
        assert_eq!(
            exact_count(&engine, "המלך", &["/author/רשי", "/era/אחרונים"]),
            0
        );
        // ספרי יסוד.
        assert_eq!(exact_count(&engine, "המלך", &["/base"]), 1);
        // קטגוריות בלבד — ההתנהגות הישנה (OR בין קטגוריות).
        assert_eq!(exact_count(&engine, "המלך", &["/root/x", "/root/y"]), 3);
    }

    // ── Result grouping ────────────────────────────────────────────────────

    #[test]
    fn grouping_same_section_collapses_with_count() {
        let (mut engine, _dir) = make_engine();
        let text = "<h1>פרק א\n\
                    המלך דוד אמר שירה גדולה מאוד בלילה\n\
                    ועוד אמר המלך דוד דברי שירה נפלאים מאוד\n\
                    <h1>פרק ב\n\
                    המלך דוד לא אמר כאן דבר"
            .to_string();
        engine
            .add_text_book(
                "תהלים".to_string(),
                "/root".to_string(),
                "book:a".to_string(),
                0,
                0,
                text,
                None,
            )
            .unwrap();
        engine.commit().unwrap();

        let page = engine
            .search_and_count_exact(
                "שירה".to_string(),
                vec![],
                50,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                Some(ResultGrouping::SameSection),
            )
            .unwrap();
        // שתי שורות תואמות באותו סעיף — קבוצה אחת, מונה 2, אחות אחת.
        assert_eq!(page.total_count, 2, "raw hit count");
        assert_eq!(page.group_count, Some(1));
        assert_eq!(page.results.len(), 1);
        let rep = &page.results[0];
        assert_eq!(rep.merged_count, 2);
        assert_eq!(rep.merged.len(), 1);
        assert_eq!(rep.merged[0].title, "תהלים");
        // הנציג הוא המוקדם בסדר הקטלוג.
        assert!(rep.segment < rep.merged[0].segment);
        // בלי קיבוץ — שתי תוצאות שטוחות.
        let flat = engine
            .search_and_count_exact(
                "שירה".to_string(),
                vec![],
                50,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                None,
            )
            .unwrap();
        assert_eq!(flat.group_count, None);
        assert_eq!(flat.results.len(), 2);
        assert!(flat.results.iter().all(|r| r.merged_count == 1));
    }

    #[test]
    fn grouping_identical_text_merges_across_books() {
        let (mut engine, _dir) = make_engine();
        // אותה שורה (ארוכה מסף החתימה) בשני ספרים; שורה שונה בספר שלישי.
        let mishna = "אמר רבי עקיבא כל ישראל יש להם חלק לעולם הבא";
        add(&mut engine, (1u64 << 32) + 1, mishna, "book:a");
        add(&mut engine, (2u64 << 32) + 1, mishna, "book:b");
        add(
            &mut engine,
            (3u64 << 32) + 1,
            "רבי עקיבא היה דורש כתרי אותיות של תורה",
            "book:c",
        );
        engine.commit().unwrap();

        let page = engine
            .search_and_count_exact(
                "עקיבא".to_string(),
                vec![],
                50,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                Some(ResultGrouping::IdenticalText),
            )
            .unwrap();
        assert_eq!(page.total_count, 3);
        assert_eq!(page.group_count, Some(2));
        // הרחק מתקרת הקבוצות — הדגל לא מורם.
        assert!(!page.truncated);
        assert_eq!(page.results.len(), 2);
        let merged = page
            .results
            .iter()
            .find(|r| r.merged_count == 2)
            .expect("merged group");
        assert_eq!(merged.merged.len(), 1);
        // הנציג והאחות מספרים שונים.
        assert_ne!(merged.file_path, merged.merged[0].file_path);

        // דפדוף נספר בקבוצות: עמוד שני של קבוצה-אחת-לכל-עמוד.
        let second = engine
            .search_and_count_exact(
                "עקיבא".to_string(),
                vec![],
                1,
                1,
                ResultsOrder::Catalogue,
                false,
                false,
                Some(ResultGrouping::IdenticalText),
            )
            .unwrap();
        assert_eq!(second.group_count, Some(2));
        assert_eq!(second.results.len(), 1);
    }

    #[test]
    fn grouping_merges_across_segments_through_the_shared_accumulator() {
        let (mut engine, _dir) = make_engine();
        // commit בין הספרים ⇒ סגמנטים נפרדים; הצבירה המשותפת חייבת לאחד
        // את הקבוצה חוצת-הסגמנטים ולמנות את שני חבריה.
        let mishna = "אמר רבי עקיבא כל ישראל יש להם חלק לעולם הבא";
        add(&mut engine, (1u64 << 32) + 1, mishna, "book:a");
        engine.commit().unwrap();
        add(&mut engine, (2u64 << 32) + 1, mishna, "book:b");
        add(
            &mut engine,
            (3u64 << 32) + 1,
            "רבי עקיבא היה דורש כתרי אותיות של תורה",
            "book:c",
        );
        engine.commit().unwrap();

        let page = engine
            .search_and_count_exact(
                "עקיבא".to_string(),
                vec![],
                50,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                Some(ResultGrouping::IdenticalText),
            )
            .unwrap();
        assert_eq!(page.total_count, 3);
        assert_eq!(page.group_count, Some(2));
        assert!(!page.truncated);
        // הקבוצה המאוחדת נושאת את שני החברים משני הסגמנטים, והנציג הוא
        // בעל ה-id הנמוך (מהסגמנט הראשון).
        let merged = page
            .results
            .iter()
            .find(|r| r.merged_count == 2)
            .expect("merged group");
        assert_eq!(merged.id, (1u64 << 32) + 1);
        assert_eq!(merged.merged.len(), 1);
    }

    #[test]
    fn grouping_identical_text_skips_short_lines() {
        let (mut engine, _dir) = make_engine();
        // שורה זהה קצרה מסף 12 האותיות — לעולם לא מתאחדת.
        add(&mut engine, (1u64 << 32) + 1, "אמר רבא", "book:a");
        add(&mut engine, (2u64 << 32) + 1, "אמר רבא", "book:b");
        engine.commit().unwrap();

        let page = engine
            .search_and_count_exact(
                "רבא".to_string(),
                vec![],
                50,
                0,
                ResultsOrder::Catalogue,
                false,
                false,
                Some(ResultGrouping::IdenticalText),
            )
            .unwrap();
        assert_eq!(page.total_count, 2);
        assert_eq!(page.group_count, Some(2));
        assert!(page.results.iter().all(|r| r.merged_count == 1));
    }

    #[test]
    fn line_dedup_hash_ignores_punctuation_and_spacing() {
        let a = line_dedup_hash("אמר רבי עקיבא, כל ישראל יש להם חלק!");
        let b = line_dedup_hash("אמר  רבי עקיבא כל ישראל — יש להם חלק");
        assert_ne!(a, 0);
        assert_eq!(a, b);
        // שינוי אות משנה חתימה.
        assert_ne!(a, line_dedup_hash("אמר רבי עקיבה כל ישראל יש להם חלק"));
        // קצר מדי — אין חתימה.
        assert_eq!(line_dedup_hash("אמר רבא"), 0);
    }

    #[test]
    fn line_dedup_hash_distinguishes_digits_and_latin() {
        // שורות שנבדלות רק במספר אינן כפילות.
        let a = line_dedup_hash("יש לשלם מאה שקלים עד יום 15 בחודש");
        let b = line_dedup_hash("יש לשלם מאה שקלים עד יום 16 בחודש");
        assert_ne!(a, 0);
        assert_ne!(a, b);
        // וכך גם אותיות לטיניות...
        assert_ne!(
            line_dedup_hash("ראו במהדורת ניו יורק עמוד A דפוס ראשון"),
            line_dedup_hash("ראו במהדורת ניו יורק עמוד B דפוס ראשון"),
        );
        // ...אבל רישיות לטיניות מקופלות, כמו שאר הקיפול הטיפוגרפי.
        assert_eq!(
            line_dedup_hash("ראו במהדורת ניו יורק עמוד a דפוס ראשון"),
            line_dedup_hash("ראו במהדורת ניו יורק עמוד A דפוס ראשון"),
        );
        // אלפאנומרי משתתף בחתימה אך לא נספר לסף האותיות העבריות.
        assert_eq!(line_dedup_hash("1234567890 abcdef אמר רבא"), 0);
        // אלפאנומרי שאינו ASCII משתתף גם הוא: ספרות ערביות-הודיות ואותיות
        // לטיניות עם סימנים מבדילים שורות, בקיפול רישיות יוניקודי.
        assert_ne!(
            line_dedup_hash("יש לשלם מאה שקלים עד יום ١٥ בחודש"),
            line_dedup_hash("יש לשלם מאה שקלים עד יום ١٦ בחודש"),
        );
        assert_ne!(
            line_dedup_hash("ראו במהדורת פריז עמוד é דפוס ראשון"),
            line_dedup_hash("ראו במהדורת פריז עמוד è דפוס ראשון"),
        );
        assert_eq!(
            line_dedup_hash("ראו במהדורת פריז עמוד É דפוס ראשון"),
            line_dedup_hash("ראו במהדורת פריז עמוד é דפוס ראשון"),
        );
    }

    #[test]
    fn bounded_groups_cap_keeps_counting_existing_groups() {
        // מילוי התקרה; קבוצה קיימת ממשיכה לצבור גם בתקרה, וקבוצות גרועות
        // מכל הקיימות נדחות עם דגל.
        let mut bounded = BoundedGroups::new();
        let cap = GROUP_COLLECTOR_MAX_GROUPS as u64;
        for i in 0..cap {
            bounded.add_entry((1u8, i), (i, i, DocAddress::new(0, i as u32)));
        }
        bounded.add_entry((1u8, 0), (0, 0, DocAddress::new(1, 0)));
        for i in cap..cap + 10 {
            bounded.add_entry((1u8, i), (i, i, DocAddress::new(2, (i % 1000) as u32)));
        }
        assert!(bounded.truncated);
        assert_eq!(bounded.groups.len(), GROUP_COLLECTOR_MAX_GROUPS);
        // קבוצה קיימת ממשיכה להיספר גם בתקרה — המונה שלה מדויק.
        assert_eq!(bounded.groups[&(1u8, 0)].count, 2);
    }

    #[test]
    fn bounded_groups_keep_the_best_groups_in_any_arrival_order() {
        // מילוי התקרה בקבוצות "גרועות" (sort גבוה); קבוצה טובה שמגיעה
        // אחריהן חייבת להיכנס על חשבון הגרועה ביותר — סדר הסריקה בין
        // הסגמנטים אינו קובע אילו קבוצות שורדות.
        let mut bounded = BoundedGroups::new();
        let cap = GROUP_COLLECTOR_MAX_GROUPS as u64;
        let high_base = 1_000_000u64;
        for i in high_base..high_base + cap {
            bounded.add_entry((1u8, i), (i, i, DocAddress::new(0, (i % 1000) as u32)));
        }
        bounded.add_entry((1u8, 1), (1, 1, DocAddress::new(1, 1)));
        assert!(bounded.truncated);
        assert_eq!(bounded.groups.len(), GROUP_COLLECTOR_MAX_GROUPS);
        // הטובה נכנסה, הגרועה ביותר פונתה.
        assert!(bounded.groups.contains_key(&(1u8, 1)));
        assert!(!bounded.groups.contains_key(&(1u8, high_base + cap - 1)));
        // והעמוד הראשון מתחיל בקבוצה הטובה ביותר.
        let hits = GroupedHits {
            raw_total: cap as u32 + 1,
            truncated: bounded.truncated,
            groups: bounded.groups,
        };
        let page = SearchEngine::finalize_grouped(hits, 1, 0);
        assert_eq!(page.reps[0].id, 1);
        assert!(page.truncated);
    }

    #[test]
    fn accumulate_section_counts_respects_the_budget() {
        let mut counts: HashMap<u64, usize> = HashMap::new();
        // המילה הראשונה ממלאה את התקציב; סעיף חדש מהמילה השנייה נשמט עם
        // דגל, אבל סעיף שכבר נספר ממשיך להצטבר.
        let truncated = accumulate_section_counts(&mut counts, (0..3u64).collect(), 3);
        assert!(!truncated);
        let truncated = accumulate_section_counts(&mut counts, HashSet::from([1u64, 99]), 3);
        assert!(truncated);
        assert_eq!(counts.len(), 3);
        assert_eq!(counts[&1], 2);
        assert!(!counts.contains_key(&99));
    }

    #[test]
    fn bounded_groups_add_entry_evicts_worst_group() {
        // מסלול האיסוף פר-סגמנט: מילוי התקרה במסמכים "גרועים", ואז מסמך
        // טוב — הקבוצה שלו נכנסת; מסמך גרוע מהגרועה ביותר — נדחה.
        let mut bounded = BoundedGroups::new();
        for i in 0..GROUP_COLLECTOR_MAX_GROUPS as u64 {
            let sort = 1000 + i;
            bounded.add_entry((1u8, sort), (sort, sort, DocAddress::new(0, i as u32)));
        }
        assert!(!bounded.truncated);
        bounded.add_entry((1u8, 5), (5, 5, DocAddress::new(0, 1)));
        assert!(bounded.truncated);
        assert!(bounded.groups.contains_key(&(1u8, 5)));
        let worst_key = (1u8, 1000 + GROUP_COLLECTOR_MAX_GROUPS as u64 - 1);
        assert!(!bounded.groups.contains_key(&worst_key));
        // מסמך לקבוצה קיימת נצבר גם בתקרה, והמונה מדויק.
        bounded.add_entry((1u8, 5), (6, 6, DocAddress::new(0, 2)));
        assert_eq!(bounded.groups[&(1u8, 5)].count, 2);
        // קבוצה חדשה גרועה מכולן — נדחית.
        bounded.add_entry((1u8, 999_999), (999_999, 999_999, DocAddress::new(0, 3)));
        assert!(!bounded.groups.contains_key(&(1u8, 999_999)));
        assert_eq!(bounded.groups.len(), GROUP_COLLECTOR_MAX_GROUPS);
    }
}
