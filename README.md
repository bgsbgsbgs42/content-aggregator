# Content Aggregator

A powerful all-in-one Python tool that combines content collection, natural language processing, deduplication, adaptive web scraping, and a comprehensive command-line interface.

## Features

### Content Collection
- **RSS Feed Support**: Automatically fetch and parse articles from RSS feeds
- **Adaptive Web Scraping**: Extract content from websites with self-improving techniques
- **Source Management**: Configure and manage multiple content sources
- **Rate Limiting**: Respectful scraping with configurable rate limits
- **Robots.txt Compliance**: Respect website scraping policies

### Natural Language Processing
- **Sentiment Analysis**: Analyze and categorize content by sentiment (positive, neutral, negative)
- **Topic Modeling**: Extract and identify key topics within content
- **Text Summarization**: Automatically generate concise summaries
- **Keyword Filtering**: Focus on content matching specific keywords or topics

### Content Deduplication
- **Exact Duplicate Detection**: Identify and filter exact content duplicates
- **Similarity Grouping**: Group similar articles together
- **Content Fingerprinting**: Generate content hashes for efficient comparison
- **Configurable Thresholds**: Adjust similarity detection sensitivity

### Adaptive Scraping
- **Self-healing Extraction**: Automatically recover from failed extraction attempts
- **Selector Learning**: Improve extraction patterns over time
- **Pattern Discovery**: Automatically discover new extraction patterns
- **Extraction Quality Assessment**: Evaluate the quality of extracted content
- **Machine Learning Integration**: Train models to improve extraction quality

### Command-Line Interface
- **Comprehensive Commands**: Full access to all features through CLI
- **Filtering Options**: Flexible content filtering by various criteria
- **Export Capabilities**: Export content in multiple formats
- **Health Monitoring**: Check the status and health of content sources
- **Detailed Reporting**: Generate statistics and performance reports

## Installation

### Prerequisites
- Python 3.6 or higher
- pip (Python package manager)

### Dependencies
The Content Aggregator requires several external libraries:

```bash
pip install requests beautifulsoup4 feedparser nltk scikit-learn numpy joblib
```

NLTK data downloads will be attempted automatically when you first run the program, but you can manually download them with:

```bash
python -m nltk.downloader punkt stopwords vader_lexicon
```

### Setup
1. Clone or download this repository
2. Install the required dependencies
3. Create a configuration file (optional, see Configuration section)
4. Run the application

## Usage

The Content Aggregator provides a command-line interface with multiple subcommands for different operations.

### Basic Commands

```bash
# Update content from all sources
python content_aggregator.py update

# List content with optional filters
python content_aggregator.py list --limit 10 --sentiment positive

# Add a new source
python content_aggregator.py add --name "Example News" --url "https://example.com/feed" --type rss
```

### Content Management

```bash
# List content matching specific criteria
python content_aggregator.py list --categories news,tech --keywords python,data
python content_aggregator.py list --sentiment positive --source 2
python content_aggregator.py list --min-sentiment 0.5 --similar-to 123

# Export content to different formats
python content_aggregator.py export --output articles.json --format json
python content_aggregator.py export --output articles.csv --format csv
python content_aggregator.py export --output articles.html --format html
```

### NLP Operations

```bash
# Get sentiment statistics
python content_aggregator.py nlp sentiment

# Extract top topics
python content_aggregator.py nlp topics --limit 10

# Generate or show summaries
python content_aggregator.py nlp summarize --id 123
python content_aggregator.py nlp summarize --limit 10

# Process NLP on existing content
python content_aggregator.py nlp process --force
```

### Deduplication

```bash
# Run deduplication process
python content_aggregator.py dedup run

# View deduplication statistics
python content_aggregator.py dedup stats

# Find articles similar to a specific article
python content_aggregator.py dedup similar --id 123
```

### Adaptive Scraping Management

```bash
# Check source status and scraping performance
python content_aggregator.py adaptive status
python content_aggregator.py adaptive status --id 2

# Retrain selectors for a specific source
python content_aggregator.py adaptive retrain --id 3

# Analyze source health and get recommendations
python content_aggregator.py adaptive health

# Update robots.txt cache
python content_aggregator.py adaptive robots

# Train machine learning models
python content_aggregator.py adaptive train
```

## Configuration

The Content Aggregator can be configured using a JSON configuration file. If no configuration file is provided, default settings will be used.

### Configuration File Example

```json
{
  "sources": [
    {
      "name": "TechNews",
      "type": "rss",
      "url": "https://technews.com/feed",
      "categories": ["tech", "news"]
    },
    {
      "name": "DataScience Blog",
      "type": "webpage",
      "url": "https://datascience-blog.com",
      "article_selector": "article.post",
      "title_selector": "h1.entry-title",
      "content_selector": "div.entry-content",
      "categories": ["data science", "programming"]
    }
  ],
  "update_interval": 3600,
  "max_articles_per_source": 10,
  "keywords": ["python", "data science", "machine learning", "web development"],
  "nlp_settings": {
    "enabled": true,
    "num_topics": 5,
    "top_words_per_topic": 8,
    "summary_sentences": 3,
    "min_content_length": 100
  },
  "deduplication": {
    "enabled": true,
    "similarity_threshold": 0.75,
    "exact_duplicates_only": false,
    "title_weight": 0.3,
    "content_weight": 0.7,
    "check_within_days": 7
  },
  "adaptive_scraping": {
    "enabled": true,
    "learn_patterns": true,
    "selector_feedback": true,
    "fallback_to_generic": true,
    "respect_robots_txt": true,
    "default_rate_limit": 5
  }
}
```

### Command-Line Configuration

You can specify a configuration file and database path when running the tool:

```bash
python content_aggregator.py --config my_config.json --db content_database.db [command]
```

## Database

The Content Aggregator uses SQLite to store content and metadata. The database includes tables for:

- Sources: Information about content sources
- Content: The actual articles and their metadata
- Keywords: Keywords for content filtering
- Similarity groups: Groups of similar content
- Scrape patterns: Extraction patterns for each source
- Training data: Data for improving extraction quality

## Advanced Usage

### Managing Sources

The health of your content sources can be monitored and improved:

```bash
# Check source health and get recommendations
python content_aggregator.py adaptive health

# View detailed source status
python content_aggregator.py adaptive status --id 2

# Retrain a problematic source
python content_aggregator.py adaptive retrain --id 3
```

### Analyzing Content

Generate insights from your collected content:

```bash
# Get sentiment distribution
python content_aggregator.py nlp sentiment

# Find top topics across all content
python content_aggregator.py nlp topics --limit 15

# Find duplicates and similar content
python content_aggregator.py dedup stats
```

## Extending the Tool

The Content Aggregator is designed to be extensible:

- Add new sources in the configuration file
- Create custom extraction patterns for specific sites
- Adjust NLP settings to better match your content
- Tune deduplication parameters for optimal grouping

## Troubleshooting

### Common Issues

- **Missing Dependencies**: Make sure you've installed all required libraries
- **Extraction Failures**: Run `adaptive status` to check source health
- **High Duplicate Rate**: Adjust similarity threshold in configuration
- **Performance Issues**: Reduce max_articles_per_source or increase update_interval

### Logs

The Content Aggregator writes logs to `content_aggregator.log` that can help diagnose issues.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Beautiful Soup for HTML parsing
- NLTK for natural language processing capabilities
- scikit-learn for machine learning features
- feedparser for RSS feed handling
