# Content Aggregator

A powerful Python-based content aggregator tool with web interface, NLP capabilities, and adaptive scraping.

## Features

### Core Functionality
- **Content Collection**: Aggregate content from various sources (RSS feeds and webpages)
- **Smart Filtering**: Filter content by categories, keywords, sentiment, and more
- **Export Options**: Export content in JSON, CSV, or HTML formats

### NLP Capabilities
- **Sentiment Analysis**: Categorize content as positive, negative, or neutral
- **Topic Modeling**: Automatically identify and tag main topics
- **Text Summarization**: Generate concise summaries of longer articles

### Content Deduplication
- **Exact Duplicate Detection**: Identify and filter duplicate content
- **Similarity Grouping**: Group similar articles from different sources
- **Customizable Thresholds**: Configure similarity sensitivity

### Adaptive Scraping
- **Self-Healing Scraper**: Adapt to website changes automatically
- **Pattern Learning**: Continuously improve content extraction
- **Ethical Web Crawling**: Respect robots.txt and implement rate limiting

### Web Interface
- **User Authentication**: Personal accounts with preferences
- **Interactive Dashboard**: Visualize content trends and statistics
- **Bookmarking System**: Save articles and add personal notes
- **Reading History**: Track read articles for better recommendations
- **Admin Controls**: Manage sources, users, and system settings

## Requirements

- Python 3.6+
- Required libraries:
  - Flask
  - SQLAlchemy
  - BeautifulSoup4
  - Requests
  - NLTK
  - scikit-learn
  - pandas
  - Plotly
  - feedparser

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/content-aggregator.git
   cd content-aggregator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run initial setup:
   ```
   python setup.py
   ```

4. Create or modify the configuration file (`config.json`):
   ```json
   {
     "sources": [
       {
         "name": "Example News",
         "type": "rss",
         "url": "https://example.com/feed",
         "categories": ["news", "technology"]
       }
     ],
     "nlp_settings": {
       "enabled": true,
       "num_topics": 3,
       "summary_sentences": 3
     },
     "deduplication": {
       "enabled": true,
       "similarity_threshold": 0.75
     },
     "adaptive_scraping": {
       "enabled": true,
       "respect_robots_txt": true,
       "default_rate_limit": 5
     }
   }
   ```

## Usage

### Command Line Interface

```bash
# Update content from all sources
python content_aggregator.py update

# List recent content
python content_aggregator.py list --limit 10

# Export content to a file
python content_aggregator.py export --output content.html --format html

# Search content
python content_aggregator.py list --keywords "python,data" --sentiment positive

# Run NLP analysis
python content_aggregator.py nlp sentiment
python content_aggregator.py nlp topics

# Manage deduplication
python content_aggregator.py dedup run
python content_aggregator.py dedup stats

# Adaptive scraping commands
python content_aggregator.py adaptive status
python content_aggregator.py adaptive health
```

### Web Interface

```bash
# Start the web interface
python run.py
```

Then open a browser and navigate to http://localhost:5000

## Web Interface Features

- **Homepage**: View recent content customized to your preferences
- **Search**: Advanced search functionality with multiple filters
- **Dashboard**: Interactive charts showing content trends
- **User Features**:
  - Register and login
  - Set content preferences
  - Bookmark articles
  - View reading history
- **Admin Features**:
  - Manage content sources
  - Trigger content updates
  - Monitor system health
  - Manage users

## Customization

### Adding a Source

```bash
# Add an RSS feed
python content_aggregator.py add --name "Tech News" --url "https://technews.com/feed" --type rss

# Add a webpage to scrape
python content_aggregator.py add --name "Tech Blog" --url "https://techblog.com" --type webpage
```

### Configure Source Scraping

For webpage sources, you can specify CSS selectors in the configuration file:

```json
{
  "sources": [
    {
      "name": "Tech Blog",
      "type": "webpage",
      "url": "https://techblog.com",
      "article_selector": "article.post",
      "title_selector": "h1.title",
      "content_selector": "div.content"
    }
  ]
}
```

## Extending

### Adding New Source Types

The system is designed to be extensible. To add support for a new source type:

1. Add a new method to the `ContentAggregator` class in `content_aggregator.py`
2. Update the command-line interface in the `main()` function
3. Add any necessary configuration options to the config file

### Adding New NLP Features

To add new NLP capabilities:

1. Add methods to the `ContentAggregator` class
2. Update the database schema to store new metadata
3. Extend the CLI and web interface to expose the new features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


