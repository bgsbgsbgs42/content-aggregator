#!/usr/bin/env python3
"""
Content Aggregator - All-in-One Tool (Asynchronous Version)

This script combines content collection, NLP processing, deduplication, 
adaptive scraping, and CLI functionality in a single file using asyncio.
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
import aiosqlite
import aiohttp
from bs4 import BeautifulSoup
import feedparser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse, urljoin
from collections import defaultdict, Counter
from heapq import nlargest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("content_aggregator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.warning(f"Unable to download NLTK data: {e}")

class ContentAggregator:
    """A tool to collect content from various web sources with advanced features."""
    
    def __init__(self, config_file=None, db_path="content.db"):
        """
        Initialize the content aggregator.
        
        Args:
            config_file (str): Path to configuration file
            db_path (str): Path to SQLite database file
        """
        self.config = self._load_config(config_file)
        self.db_path = db_path
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = None
        self.robots_cache = {}
        self.rate_limit_locks = {}
        self.access_queues = {}
        self._initialize_nlp()

    def _load_config(self, config_file):
        """Load configuration from file or use default configuration."""
        default_config = {
            "sources": [
                {
                    "name": "Example News",
                    "type": "rss",
                    "url": "https://example.com/feed",
                    "categories": ["news", "technology"]
                }
            ],
            "update_interval": 3600,
            "max_articles_per_source": 10,
            "keywords": ["python", "data science", "web development"],
            "nlp_settings": {
                "enabled": True,
                "num_topics": 3,
                "top_words_per_topic": 5,
                "summary_sentences": 3,
                "min_content_length": 100
            },
            "deduplication": {
                "enabled": True,
                "similarity_threshold": 0.75,
                "exact_duplicates_only": False,
                "title_weight": 0.3,
                "content_weight": 0.7,
                "use_min_hash": True,
                "check_within_days": 7,
                "keep_newest": True
            },
            "adaptive_scraping": {
                "enabled": True,
                "learn_patterns": True,
                "selector_feedback": True,
                "auto_retry_count": 3,
                "training_sample_size": 50,
                "min_content_length": 100,
                "min_extraction_quality": 0.5,
                "fallback_to_generic": True,
                "respect_robots_txt": True,
                "default_rate_limit": 5,
                "retry_decay": 0.9,
                "model_update_frequency": 100,
                "selector_discovery": True,
                "quality_assessment": True
            }
        }
        
        if not config_file or not os.path.exists(config_file):
            logger.warning("No config file found. Using default configuration.")
            return default_config
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully.")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration.")
            return default_config

    async def _setup_database(self):
        """Set up SQLite database for storing content."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS sources (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        url TEXT UNIQUE NOT NULL,
                        source_type TEXT NOT NULL,
                        last_updated TIMESTAMP,
                        scrape_config TEXT,
                        robots_txt TEXT,
                        rate_limit INTEGER DEFAULT 5,
                        last_access TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        error_count INTEGER DEFAULT 0,
                        adaptive_config_version INTEGER DEFAULT 1
                    )
                ''')
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS content (
                        id INTEGER PRIMARY KEY,
                        source_id INTEGER,
                        title TEXT NOT NULL,
                        content TEXT,
                        url TEXT UNIQUE NOT NULL,
                        published TIMESTAMP,
                        added TIMESTAMP,
                        categories TEXT,
                        sentiment_score REAL,
                        sentiment_label TEXT,
                        topics TEXT,
                        summary TEXT,
                        content_hash TEXT,
                        similarity_group INTEGER,
                        is_duplicate INTEGER DEFAULT 0,
                        extraction_quality REAL,
                        FOREIGN KEY (source_id) REFERENCES sources (id)
                    )
                ''')
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS keywords (
                        id INTEGER PRIMARY KEY,
                        word TEXT UNIQUE NOT NULL
                    )
                ''')
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS similarity_groups (
                        id INTEGER PRIMARY KEY,
                        created TIMESTAMP,
                        updated TIMESTAMP,
                        main_article_id INTEGER,
                        article_count INTEGER DEFAULT 1,
                        FOREIGN KEY (main_article_id) REFERENCES content (id)
                    )
                ''')
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS scrape_patterns (
                        id INTEGER PRIMARY KEY,
                        source_id INTEGER,
                        selector_type TEXT,
                        selector TEXT,
                        target_element TEXT,
                        success_count INTEGER DEFAULT 0,
                        fail_count INTEGER DEFAULT 0,
                        last_used TIMESTAMP,
                        score REAL DEFAULT 0.0,
                        is_learned BOOLEAN DEFAULT 0,
                        FOREIGN KEY (source_id) REFERENCES sources (id)
                ''')
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS scrape_training_data (
                        id INTEGER PRIMARY KEY,
                        source_id INTEGER,
                        html_sample TEXT,
                        article_content TEXT,
                        article_title TEXT,
                        successful_selectors TEXT,
                        date_collected TIMESTAMP,
                        FOREIGN KEY (source_id) REFERENCES sources (id)
                ''')
                await conn.commit()
                logger.info("Database setup completed.")
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise

    def _initialize_nlp(self):
        """Initialize NLP components for text analysis."""
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
            self.punctuation = set(string.punctuation)
            self.n_topics = self.config.get("nlp_settings", {}).get("num_topics", 3)
            self.n_top_words = self.config.get("nlp_settings", {}).get("top_words_per_topic", 5)
            self.summary_sentences = self.config.get("nlp_settings", {}).get("summary_sentences", 3)
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            logger.info("NLP components initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            logger.warning("NLP features may not work properly.")

    async def update_content(self):
        """Fetch and update content from all configured sources."""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT id, name, url, source_type FROM sources")
            sources = await cursor.fetchall()
        
        tasks = [self._update_source(source) for source in sources]
        results = await asyncio.gather(*tasks)
        
        summary = {
            "total_sources": len(sources),
            "successful_sources": sum(1 for result in results if result["success"]),
            "failed_sources": sum(1 for result in results if not result["success"]),
            "new_articles": sum(result["new_articles"] for result in results)
        }
        return summary

    async def _update_source(self, source):
        """Update content from a single source."""
        source_id, name, url, source_type = source
        source_info = {
            "name": name,
            "url": url,
            "type": source_type
        }
        
        for cfg_source in self.config["sources"]:
            if cfg_source["url"] == url:
                source_info.update(cfg_source)
                break
        
        try:
            if source_type == "rss":
                articles = await self.fetch_rss_feed(source_info)
            elif source_type == "webpage":
                articles = await self.fetch_webpage(source_info)
            else:
                logger.warning(f"Unknown source type '{source_type}' for {name}")
                return {"success": False, "new_articles": 0}
            
            if articles:
                new_articles = await self._save_to_database(articles, source_id)
                logger.info(f"Updated {name}: {new_articles} new articles")
                return {"success": True, "new_articles": new_articles}
            else:
                return {"success": False, "new_articles": 0}
        except Exception as e:
            logger.error(f"Error updating source {name}: {e}")
            return {"success": False, "new_articles": 0}

    async def fetch_rss_feed(self, source):
        """Fetch and parse content from an RSS feed."""
        try:
            feed = feedparser.parse(source["url"])
            logger.info(f"Fetched RSS feed: {source['name']} with {len(feed.entries)} entries")
            
            articles = []
            for entry in feed.entries[:self.config["max_articles_per_source"]]:
                article = {
                    "title": entry.get("title", "No title"),
                    "content": entry.get("summary", entry.get("description", "")),
                    "url": entry.get("link", ""),
                    "published": entry.get("published_parsed") or entry.get("updated_parsed"),
                    "categories": [tag.term for tag in entry.get("tags", [])] if hasattr(entry, "tags") else []
                }
                
                if article["published"]:
                    article["published"] = time.strftime('%Y-%m-%d %H:%M:%S', article["published"])
                else:
                    article["published"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                articles.append(article)
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching RSS feed {source['url']}: {e}")
            return []

    async def fetch_webpage(self, source):
        """Scrape content from a webpage with adaptive techniques."""
        try:
            adaptive_scraping = self.config.get("adaptive_scraping", {}).get("enabled", True)
            source_id = await self._get_or_create_source_id(source)
            
            if not await self._check_robots_txt(source["url"]):
                logger.warning(f"Robots.txt disallows scraping {source['url']}")
                return []
            
            await self._rate_limit_request(source["url"])
            async with self.session.get(source["url"], headers=self.headers, timeout=10) as response:
                response.raise_for_status()
                html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            articles = []
            
            if adaptive_scraping:
                articles = await self._adaptive_scrape(soup, source, source_id)
            else:
                articles = await self._traditional_scrape(soup, source)
            
            logger.info(f"Fetched webpage: {source['name']} with {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"Error fetching webpage {source['url']}: {e}")
            await self._update_source_success_rate_on_error(source["url"])
            return []

    async def _adaptive_scrape(self, soup, source, source_id):
        """Perform adaptive scraping using learned selectors."""
        articles = []
        source_selectors = await self._get_source_selectors(source_id)
        title_selectors = source_selectors.get("title", [])
        content_selectors = source_selectors.get("content", [])
        
        title, title_selector, title_quality = await self._extract_with_selectors(soup, title_selectors, "title")
        content, content_selector, content_quality = await self._extract_with_selectors(soup, content_selectors, "content")
        
        if title_selector:
            await self._update_selector_score(source_id, title_selector, "title", title is not None)
        if content_selector:
            await self._update_selector_score(source_id, content_selector, "content", content is not None)
        
        if title and content:
            successful_selectors = {"title": title_selector, "content": content_selector}
            new_selectors = await self._discover_new_selectors(soup, successful_selectors)
            await self._save_new_selectors(source_id, new_selectors)
            await self._save_training_data(source_id, str(soup), content, title, successful_selectors)
            
            article = {
                "title": title,
                "content": content,
                "url": source["url"],
                "published": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "categories": source.get("categories", []),
                "extraction_quality": min(title_quality, content_quality)
            }
            articles.append(article)
            await self._update_source_success_rate(source_id, True)
        else:
            if self.config.get("adaptive_scraping", {}).get("fallback_to_generic", True):
                articles = await self._traditional_scrape(soup, source)
                if articles:
                    await self._update_source_success_rate(source_id, True)
                else:
                    await self._update_source_success_rate(source_id, False)
        
        return articles

    async def _traditional_scrape(self, soup, source):
        """Perform traditional scraping using predefined selectors."""
        articles = []
        article_selector = source.get("article_selector", "article")
        title_selector = source.get("title_selector", "h2")
        content_selector = source.get("content_selector", "p")
        link_selector = source.get("link_selector", "a")
        
        for article_elem in soup.select(article_selector)[:self.config["max_articles_per_source"]]:
            title_elem = article_elem.select_one(title_selector)
            content_elem = article_elem.select_one(content_selector)
            link_elem = article_elem.select_one(link_selector)
            
            title = title_elem.text.strip() if title_elem else "No title"
            content = content_elem.text.strip() if content_elem else ""
            
            url = ""
            if link_elem and link_elem.has_attr('href'):
                url = link_elem['href']
                if not urlparse(url).netloc:
                    parsed_source_url = urlparse(source["url"])
                    base_url = f"{parsed_source_url.scheme}://{parsed_source_url.netloc}"
                    url = base_url + ('' if url.startswith('/') else '/') + url
            else:
                url = source["url"]
            
            article = {
                "title": title,
                "content": content,
                "url": url,
                "published": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "categories": source.get("categories", [])
            }
            articles.append(article)
        
        return articles

    async def _get_or_create_source_id(self, source):
        """Get or create a source ID in the database."""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT id FROM sources WHERE url = ?", (source["url"],))
            result = await cursor.fetchone()
            
            if result:
                return result[0]
            else:
                await conn.execute(
                    "INSERT INTO sources (name, url, source_type) VALUES (?, ?, ?)",
                    (source.get("name", "Unknown"), source["url"], source.get("type", "webpage"))
                )
                await conn.commit()
                return cursor.lastrowid

    async def _check_robots_txt(self, url):
        """Check if scraping is allowed by robots.txt."""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url in self.robots_cache:
            return self.robots_cache[base_url]
        
        try:
            async with self.session.get(f"{base_url}/robots.txt") as response:
                if response.status == 200:
                    robots_txt = await response.text()
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(robots_txt.splitlines())
                    self.robots_cache[base_url] = parser.can_fetch(self.headers["User-Agent"], parsed_url.path or "/")
                    return self.robots_cache[base_url]
                else:
                    self.robots_cache[base_url] = True
                    return True
        except Exception as e:
            logger.error(f"Error checking robots.txt for {base_url}: {e}")
            self.robots_cache[base_url] = True
            return True

    async def _rate_limit_request(self, url):
        """Apply rate limiting for requests."""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url not in self.rate_limit_locks:
            self.rate_limit_locks[base_url] = asyncio.Lock()
            self.access_queues[base_url] = asyncio.Queue()
        
        async with self.rate_limit_locks[base_url]:
            await self.access_queues[base_url].put(None)
            await asyncio.sleep(60 / self.config.get("adaptive_scraping", {}).get("default_rate_limit", 5))
            await self.access_queues[base_url].get()

    async def _extract_with_selectors(self, soup, selectors, element_type):
        """Extract content using a list of selectors."""
        for selector in selectors:
            try:
                if selector["selector_type"] == "css":
                    elements = soup.select(selector["selector"])
                elif selector["selector_type"] == "xpath":
                    elements = soup.find_all(selector["selector"].split("/")[-1])
                else:
                    continue
                
                if not elements:
                    continue
                
                if element_type == "title":
                    content = elements[0].get_text().strip()
                elif element_type == "content":
                    content = "\n".join([elem.get_text().strip() for elem in elements])
                else:
                    content = "\n".join([elem.get_text().strip() for elem in elements])
                
                if len(content) >= self.config.get("adaptive_scraping", {}).get("min_content_length", 100):
                    quality_score = await self._assess_extraction_quality(content, element_type)
                    return content, selector, quality_score
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")
                continue
        
        return None, None, 0.0

    async def _assess_extraction_quality(self, content, element_type):
        """Assess the quality of extracted content."""
        if not content:
            return 0.0
        
        try:
            if self.extraction_quality_model and element_type == "content":
                features = [
                    len(content),
                    len(content.split()),
                    len(content.split('\n')),
                    sum(1 for c in content if c.isupper()) / len(content) if len(content) > 0 else 0,
                    sum(1 for c in content if c in string.punctuation) / len(content) if len(content) > 0 else 0,
                    len(set(content.split())) / len(content.split()) if len(content.split()) > 0 else 0,
                ]
                quality_score = float(self.extraction_quality_model.predict_proba([features])[0][1])
                return quality_score
            
            if element_type == "title":
                words = content.split()
                if len(words) < 3 or len(words) > 30:
                    return 0.5
                if content.islower():
                    return 0.6
                if re.search(r'\b(lorem|ipsum|dolor)\b', content.lower()):
                    return 0.1
                return 0.9
            
            elif element_type == "content":
                words = content.split()
                if len(words) < 20:
                    return 0.3
                paragraphs = content.split('\n\n')
                if len(paragraphs) < 2:
                    return 0.7
                if re.search(r'\b(lorem|ipsum|dolor)\b', content.lower()):
                    return 0.1
                html_ratio = len(re.findall(r'<[^>]+>', content)) / len(content) if len(content) > 0 else 1
                if html_ratio > 0.1:
                    return 0.4
                sentences = sent_tokenize(content)
                avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
                if avg_sent_len < 3 or avg_sent_len > 40:
                    return 0.6
                return 0.85
            
            else:
                return 0.7
        except Exception as e:
            logger.error(f"Error assessing extraction quality: {e}")
            return 0.5

    async def _update_selector_score(self, source_id, selector, target_element, success=True):
        """Update the score of a selector based on success or failure."""
        if not self.config.get("adaptive_scraping", {}).get("selector_feedback", True):
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    """
                    SELECT id, success_count, fail_count, score
                    FROM scrape_patterns
                    WHERE source_id = ? AND selector_type = ? AND selector = ? AND target_element = ?
                    """,
                    (source_id, selector["selector_type"], selector["selector"], target_element)
                )
                result = await cursor.fetchone()
                
                if result:
                    selector_id, success_count, fail_count, old_score = result
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                    total = success_count + fail_count
                    new_score = success_count / total if total > 0 else 0.5
                    alpha = 0.8
                    smoothed_score = (alpha * new_score) + ((1 - alpha) * old_score)
                    
                    await conn.execute(
                        """
                        UPDATE scrape_patterns
                        SET success_count = ?, fail_count = ?, score = ?, last_used = ?
                        WHERE id = ?
                        """,
                        (success_count, fail_count, smoothed_score, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), selector_id)
                    )
                else:
                    initial_score = 0.7 if success else 0.3
                    success_count = 1 if success else 0
                    fail_count = 0 if success else 1
                    
                    await conn.execute(
                        """
                        INSERT INTO scrape_patterns
                        (source_id, selector_type, selector, target_element, success_count, fail_count, score, last_used)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            selector["selector_type"],
                            selector["selector"],
                            target_element,
                            success_count,
                            fail_count,
                            initial_score,
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )
                    )
                await conn.commit()
        except Exception as e:
            logger.error(f"Database error updating selector score: {e}")

    async def _discover_new_selectors(self, soup, successful_selectors):
        """Discover new potential selectors from a successful page."""
        if not self.config.get("adaptive_scraping", {}).get("selector_discovery", True):
            return {}
        
        new_selectors = defaultdict(list)
        
        try:
            if "title" in successful_selectors:
                successful_title = successful_selectors["title"]["selector"]
                successful_element = soup.select_one(successful_title)
                
                if successful_element:
                    tag_name = successful_element.name
                    classes = successful_element.get("class", [])
                    parent = successful_element.parent
                    
                    if successful_element.get("id"):
                        new_selectors["title"].append({
                            "selector_type": "css",
                            "selector": f"#{successful_element['id']}"
                        })
                    
                    for cls in classes:
                        if "title" in cls.lower() or "heading" in cls.lower() or "header" in cls.lower():
                            new_selectors["title"].append({
                                "selector_type": "css",
                                "selector": f".{cls}"
                            })
                    
                    if parent and parent.name != "body":
                        if parent.get("class"):
                            parent_class = parent.get("class")[0] if parent.get("class") else ""
                            if parent_class:
                                new_selectors["title"].append({
                                    "selector_type": "css",
                                    "selector": f".{parent_class} {tag_name}"
                                })
            
            if "content" in successful_selectors:
                successful_content = successful_selectors["content"]["selector"]
                successful_element = soup.select_one(successful_content)
                
                if successful_element:
                    tag_name = successful_element.name
                    classes = successful_element.get("class", [])
                    
                    if successful_element.get("id"):
                        new_selectors["content"].append({
                            "selector_type": "css",
                            "selector": f"#{successful_element['id']}"
                        })
                    
                    for cls in classes:
                        if "content" in cls.lower() or "article" in cls.lower() or "post" in cls.lower() or "entry" in cls.lower():
                            new_selectors["content"].append({
                                "selector_type": "css",
                                "selector": f".{cls}"
                            })
                    
                    paragraphs = successful_element.find_all("p")
                    if paragraphs and len(paragraphs) >= 3:
                        common_parents = defaultdict(int)
                        for p in paragraphs:
                            parent = p.parent
                            if parent and parent.name != "body":
                                if parent.get("class"):
                                    parent_class = parent.get("class")[0] if parent.get("class") else ""
                                    if parent_class:
                                        common_parents[f".{parent_class} p"] += 1
                        
                        if common_parents:
                            most_common = max(common_parents.items(), key=lambda x: x[1])
                            new_selectors["content"].append({
                                "selector_type": "css",
                                "selector": most_common[0]
                            })
            
            for element_type in new_selectors:
                unique_selectors = []
                seen = set()
                for selector in new_selectors[element_type]:
                    if selector["selector"] not in seen:
                        seen.add(selector["selector"])
                        unique_selectors.append(selector)
                
                new_selectors[element_type] = unique_selectors
            
            return new_selectors
        except Exception as e:
            logger.error(f"Error discovering new selectors: {e}")
            return {}

    async def _save_new_selectors(self, source_id, new_selectors):
        """Save new selectors to the database."""
        if not new_selectors:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                for element_type, selectors in new_selectors.items():
                    for selector in selectors:
                        await conn.execute(
                            """
                            INSERT OR IGNORE INTO scrape_patterns
                            (source_id, selector_type, selector, target_element, score, last_used, is_learned)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                source_id,
                                selector["selector_type"],
                                selector["selector"],
                                element_type,
                                0.5,
                                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                1
                            )
                        )
                await conn.commit()
        except Exception as e:
            logger.error(f"Database error saving new selectors: {e}")

    async def _save_training_data(self, source_id, html_sample, content, title, successful_selectors):
        """Save successful extraction as training data."""
        if not content or not title:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM scrape_training_data WHERE source_id = ?",
                    (source_id,)
                )
                count = (await cursor.fetchone())[0]
                max_samples = self.config.get("adaptive_scraping", {}).get("training_sample_size", 50)
                
                if count >= max_samples:
                    await conn.execute(
                        """
                        DELETE FROM scrape_training_data
                        WHERE id = (
                            SELECT id FROM scrape_training_data
                            WHERE source_id = ?
                            ORDER BY date_collected ASC
                            LIMIT 1
                        )
                        """,
                        (source_id,)
                    )
                
                selectors_json = json.dumps(successful_selectors)
                html_sample = html_sample[:50000] if html_sample else ""
                
                await conn.execute(
                    """
                    INSERT INTO scrape_training_data
                    (source_id, html_sample, article_content, article_title, successful_selectors, date_collected)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        html_sample,
                        content,
                        title,
                        selectors_json,
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    )
                )
                await conn.commit()
                
                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM scrape_training_data
                    WHERE date_collected > (
                        SELECT last_updated FROM sources WHERE id = ?
                    )
                    """,
                    (source_id,)
                )
                new_samples = (await cursor.fetchone())[0]
                model_update_frequency = self.config.get("adaptive_scraping", {}).get("model_update_frequency", 100)
                
                if new_samples >= model_update_frequency:
                    await self._train_extraction_models()
                    await conn.execute(
                        "UPDATE sources SET last_updated = ? WHERE id = ?",
                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), source_id)
                    )
                    await conn.commit()
        except Exception as e:
            logger.error(f"Database error saving training data: {e}")

    async def _train_extraction_models(self):
        """Train machine learning models for content extraction quality assessment."""
        try:
            if not joblib:
                logger.warning("Cannot train models: joblib module not available")
                return
            
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    """
                    SELECT article_content, successful_selectors
                    FROM scrape_training_data
                    WHERE article_content IS NOT NULL AND article_content != ''
                    """
                )
                training_data = await cursor.fetchall()
            
            if not training_data or len(training_data) < 10:
                logger.warning("Not enough training data to build models")
                return
            
            X_quality = []
            y_quality = []
            
            for content, selectors in training_data:
                if not content:
                    continue
                
                features = [
                    len(content),
                    len(content.split()),
                    len(content.split('\n')),
                    sum(1 for c in content if c.isupper()) / len(content) if len(content) > 0 else 0,
                    sum(1 for c in content if c in string.punctuation) / len(content) if len(content) > 0 else 0,
                    len(set(content.split())) / len(content.split()) if len(content.split()) > 0 else 0,
                ]
                X_quality.append(features)
                y_quality.append(1)
            
            if len(X_quality) < 10:
                logger.warning("Not enough quality content samples for training")
                return
            
            for i in range(min(len(X_quality), 30)):
                if i < len(training_data) and training_data[i][0]:
                    content = training_data[i][0]
                    
                    if random.random() < 0.33:
                        damaged = content[:int(len(content) * 0.2)]
                    elif random.random() < 0.66:
                        words = content.split()
                        random.shuffle(words)
                        damaged = " ".join(words[:int(len(words) * 0.5)])
                    else:
                        words = content.split()
                        sampled = random.choices(words, k=min(20, len(words)))
                        damaged = " ".join(sampled) * 3
                    
                    features = [
                        len(damaged),
                        len(damaged.split()),
                        len(damaged.split('\n')),
                        sum(1 for c in damaged if c.isupper()) / len(damaged) if len(damaged) > 0 else 0,
                        sum(1 for c in damaged if c in string.punctuation) / len(damaged) if len(damaged) > 0 else 0,
                        len(set(damaged.split())) / len(damaged.split()) if len(damaged.split()) > 0 else 0,
                    ]
                    X_quality.append(features)
                    y_quality.append(0)
            
            X_quality_train, X_quality_test, y_quality_train, y_quality_test = train_test_split(
                X_quality, y_quality, test_size=0.2, random_state=42
            )
            
            quality_model = RandomForestClassifier(n_estimators=50, random_state=42)
            quality_model.fit(X_quality_train, y_quality_train)
            
            y_pred = quality_model.predict(X_quality_test)
            accuracy = accuracy_score(y_quality_test, y_pred)
            f1 = f1_score(y_quality_test, y_pred)
            
            logger.info(f"Extraction quality model trained. Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
            
            try:
                model_dir = os.path.dirname(self.db_path)
                joblib.dump(quality_model, os.path.join(model_dir, "extraction_quality_model.joblib"))
                self.extraction_quality_model = quality_model
                logger.info("Models trained and saved successfully")
            except Exception as e:
                logger.error(f"Error saving trained models: {e}")
                self.extraction_quality_model = quality_model
        except Exception as e:
            logger.error(f"Error training extraction models: {e}")

    async def _update_source_success_rate(self, source_id, success=True):
        """Update the success rate for a source."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    """
                    SELECT success_rate, access_count, error_count
                    FROM sources
                    WHERE id = ?
                    """,
                    (source_id,)
                )
                result = await cursor.fetchone()
                if not result:
                    return
                
                success_rate, access_count, error_count = result
                access_count += 1
                if not success:
                    error_count += 1
                
                new_success_rate = (access_count - error_count) / access_count if access_count > 0 else 0.0
                
                await conn.execute(
                    """
                    UPDATE sources
                    SET success_rate = ?, access_count = ?, error_count = ?, last_access = ?
                    WHERE id = ?
                    """,
                    (
                        new_success_rate,
                        access_count,
                        error_count,
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        source_id
                    )
                )
                await conn.commit()
        except Exception as e:
            logger.error(f"Database error updating source success rate: {e}")

    async def _update_source_success_rate_on_error(self, url):
        """Update source success rate on error."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("SELECT id FROM sources WHERE url = ?", (url,))
                result = await cursor.fetchone()
                if result:
                    await self._update_source_success_rate(result[0], False)
        except Exception as e:
            logger.error(f"Error updating source success rate on error: {e}")

    async def get_source_status(self, source_id=None):
        """Get status information about sources and their scraping performance."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                if source_id:
                    cursor = await conn.execute(
                        """
                        SELECT s.id, s.name, s.url, s.source_type, s.success_rate, 
                               s.access_count, s.error_count, s.last_access, s.rate_limit,
                               COUNT(sp.id) as pattern_count
                        FROM sources s
                        LEFT JOIN scrape_patterns sp ON s.id = sp.source_id
                        WHERE s.id = ?
                        GROUP BY s.id
                        """,
                        (source_id,)
                    )
                else:
                    cursor = await conn.execute(
                        """
                        SELECT s.id, s.name, s.url, s.source_type, s.success_rate, 
                               s.access_count, s.error_count, s.last_access, s.rate_limit,
                               COUNT(sp.id) as pattern_count
                        FROM sources s
                        LEFT JOIN scrape_patterns sp ON s.id = sp.source_id
                        GROUP BY s.id
                        """
                    )
                results = []
                for row in await cursor.fetchall():
                    source_id, name, url, source_type, success_rate, access_count, error_count, last_access, rate_limit, pattern_count = row
                    
                    cursor2 = await conn.execute(
                        """
                        SELECT target_element, selector, score
                        FROM scrape_patterns
                        WHERE source_id = ?
                        ORDER BY score DESC
                        LIMIT 6
                        """,
                        (source_id,)
                    )
                    top_selectors = []
                    for target, selector, score in await cursor2.fetchall():
                        top_selectors.append({
                            "target": target,
                            "selector": selector,
                            "score": score
                        })
                    
                    source_info = {
                        "id": source_id,
                        "name": name,
                        "url": url,
                        "type": source_type,
                        "success_rate": success_rate,
                        "access_count": access_count,
                        "error_count": error_count,
                        "last_access": last_access,
                        "rate_limit": rate_limit,
                        "pattern_count": pattern_count,
                        "top_selectors": top_selectors,
                        "health": "good" if success_rate > 0.8 else ("fair" if success_rate > 0.5 else "poor")
                    }
                    results.append(source_info)
                return results
        except Exception as e:
            logger.error(f"Database error getting source status: {e}")
            return []

    async def retrain_source(self, source_id):
        """Force retraining of selectors for a specific source."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("SELECT name FROM sources WHERE id = ?", (source_id,))
                result = await cursor.fetchone()
                if not result:
                    return {"status": "error", "message": f"Source ID {source_id} not found"}
                
                source_name = result[0]
                await conn.execute(
                    "DELETE FROM scrape_patterns WHERE source_id = ? AND is_learned = 1",
                    (source_id,)
                )
                await conn.execute(
                    "UPDATE sources SET adaptive_config_version = adaptive_config_version + 1 WHERE id = ?",
                    (source_id,)
                )
                await conn.commit()
                return {
                    "status": "success",
                    "message": f"Retrained source '{source_name}'. New selectors will be learned during the next update."
                }
        except Exception as e:
            logger.error(f"Database error retraining source: {e}")
            return {"status": "error", "message": str(e)}

    async def analyze_source_health(self):
        """Analyze the health of all sources and recommend improvements."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    """
                    SELECT id, name, url, source_type, success_rate, access_count, error_count
                    FROM sources
                    ORDER BY success_rate ASC
                    """
                )
                sources = []
                recommendations = []
                total_sources = 0
                healthy_sources = 0
                
                for row in await cursor.fetchall():
                    source_id, name, url, source_type, success_rate, access_count, error_count = row
                    total_sources += 1
                    source_health = {
                        "id": source_id,
                        "name": name,
                        "url": url,
                        "type": source_type,
                        "success_rate": success_rate,
                        "access_count": access_count,
                        "error_count": error_count,
                        "health_status": "good" if success_rate > 0.8 else ("fair" if success_rate > 0.5 else "poor")
                    }
                    if success_rate > 0.8:
                        healthy_sources += 1
                    sources.append(source_health)
                    
                    if success_rate < 0.5 and access_count > 5:
                        cursor2 = await conn.execute(
                            """
                            SELECT COUNT(*) FROM scrape_patterns
                            WHERE source_id = ? AND score > 0.7
                            """,
                            (source_id,)
                        )
                        good_selectors = (await cursor2.fetchone())[0]
                        if good_selectors == 0:
                            recommendations.append({
                                "source_id": source_id,
                                "source_name": name,
                                "recommendation": "manual_selectors",
                                "message": f"Source '{name}' has poor extraction success rate and no good selectors. Consider adding manual selectors."
                            })
                        else:
                            recommendations.append({
                                "source_id": source_id,
                                "source_name": name,
                                "recommendation": "retrain",
                                "message": f"Source '{name}' has poor extraction success rate. Consider retraining selectors."
                            })
                
                health_summary = {
                    "total_sources": total_sources,
                    "healthy_sources": healthy_sources,
                    "health_percentage": (healthy_sources / total_sources * 100) if total_sources > 0 else 0,
                    "sources": sources,
                    "recommendations": recommendations
                }
                return health_summary
        except Exception as e:
            logger.error(f"Database error analyzing source health: {e}")
            return {"status": "error", "message": str(e)}

    async def update_robots_txt_cache(self):
        """Update the robots.txt cache for all sources."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("SELECT id, url FROM sources")
                sources = await cursor.fetchall()
                updated = 0
                failed = 0
                disallowed = 0
                
                for source_id, url in sources:
                    try:
                        parsed_url = urlparse(url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        async with self.session.get(f"{base_url}/robots.txt") as response:
                            if response.status == 200:
                                robots_txt = await response.text()
                                parser = urllib.robotparser.RobotFileParser()
                                parser.parse(robots_txt.splitlines())
                                self.robots_cache[base_url] = parser.can_fetch(self.headers["User-Agent"], parsed_url.path or "/")
                                await conn.execute(
                                    "UPDATE sources SET robots_txt = ? WHERE id = ?",
                                    (f"{base_url}/robots.txt", source_id)
                                )
                                updated += 1
                                if not self.robots_cache[base_url]:
                                    disallowed += 1
                            else:
                                self.robots_cache[base_url] = True
                                updated += 1
                    except Exception as e:
                        logger.error(f"Error updating robots.txt for {url}: {e}")
                        failed += 1
                
                await conn.commit()
                return {
                    "status": "success",
                    "updated": updated,
                    "failed": failed,
                    "disallowed": disallowed
                }
        except Exception as e:
            logger.error(f"Error updating robots.txt cache: {e}")
            return {"status": "error", "message": str(e)}

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

async def main():
    """Command line interface for ContentAggregator."""
    parser = argparse.ArgumentParser(description="Content Aggregator Tool")
    parser.add_argument("--config", help="Path to configuration file", default="config.json")
    parser.add_argument("--db", help="Path to database file", default="content.db")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update content from all sources")
    
    # Add source command
    add_parser = subparsers.add_parser("add", help="Add a new content source")
    add_parser.add_argument("--name", required=True, help="Name of the source")
    add_parser.add_argument("--url", required=True, help="URL of the source")
    add_parser.add_argument("--type", required=True, choices=["rss", "webpage"], help="Type of source")
    add_parser.add_argument("--categories", help="Categories (comma-separated)")
    
    # Remove source command
    remove_parser = subparsers.add_parser("remove", help="Remove a content source")
    remove_parser.add_argument("--id", required=True, type=int, help="ID of the source to remove")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List content")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of articles")
    list_parser.add_argument("--offset", type=int, default=0, help="Starting offset")
    list_parser.add_argument("--categories", help="Filter by categories (comma-separated)")
    list_parser.add_argument("--keywords", help="Filter by keywords (comma-separated)")
    list_parser.add_argument("--source", type=int, help="Filter by source ID")
    list_parser.add_argument("--sentiment", choices=["positive", "negative", "neutral"], help="Filter by sentiment")
    list_parser.add_argument("--min-sentiment", type=float, help="Filter by minimum sentiment score")
    list_parser.add_argument("--max-sentiment", type=float, help="Filter by maximum sentiment score")
    list_parser.add_argument("--include-duplicates", action="store_true", help="Include duplicate articles")
    list_parser.add_argument("--similar-to", type=int, help="Show articles similar to this ID")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export content to a file")
    export_parser.add_argument("--output", required=True, help="Output file path")
    export_parser.add_argument("--format", choices=["json", "csv", "html"], default="json", help="Export format")
    export_parser.add_argument("--limit", type=int, default=100, help="Maximum number of articles")
    export_parser.add_argument("--categories", help="Filter by categories (comma-separated)")
    export_parser.add_argument("--keywords", help="Filter by keywords (comma-separated)")
    export_parser.add_argument("--sentiment", choices=["positive", "negative", "neutral"], help="Filter by sentiment")
    export_parser.add_argument("--include-duplicates", action="store_true", help="Include duplicate articles")
    
    # NLP commands
    nlp_parser = subparsers.add_parser("nlp", help="NLP analysis commands")
    nlp_subparsers = nlp_parser.add_subparsers(dest="nlp_command", help="NLP command to execute")
    
    # Sentiment analysis command
    sentiment_parser = nlp_subparsers.add_parser("sentiment", help="Get sentiment statistics")
    
    # Topic modeling command
    topics_parser = nlp_subparsers.add_parser("topics", help="Get top topics")
    topics_parser.add_argument("--limit", type=int, default=10, help="Maximum number of topics to return")
    
    # Summarize command
    summarize_parser = nlp_subparsers.add_parser("summarize", help="Generate or show summaries")
    summarize_parser.add_argument("--id", type=int, help="Article ID to summarize")
    summarize_parser.add_argument("--limit", type=int, default=10, help="Maximum number of articles to show summaries for")
    
    # Process NLP command
    process_parser = nlp_subparsers.add_parser("process", help="Run NLP processing on existing content")
    process_parser.add_argument("--limit", type=int, help="Maximum number of articles to process")
    process_parser.add_argument("--force", action="store_true", help="Force reprocessing of already processed content")
    
    # Deduplication commands
    dedup_parser = subparsers.add_parser("dedup", help="Content deduplication commands")
    dedup_subparsers = dedup_parser.add_subparsers(dest="dedup_command", help="Deduplication command to execute")
    
    # Deduplicate command
    deduplicate_parser = dedup_subparsers.add_parser("run", help="Run deduplication on existing content")
    deduplicate_parser.add_argument("--force", action="store_true", help="Force reprocessing all content")
    
    # Deduplication stats command
    dedup_stats_parser = dedup_subparsers.add_parser("stats", help="Show deduplication statistics")
    
    # Similar articles command
    similar_parser = dedup_subparsers.add_parser("similar", help="Show articles similar to a given article")
    similar_parser.add_argument("--id", type=int, required=True, help="Article ID to find similarities for")
    
    # Adaptive scraping commands
    adaptive_parser = subparsers.add_parser("adaptive", help="Adaptive scraping commands")
    adaptive_subparsers = adaptive_parser.add_subparsers(dest="adaptive_command", help="Adaptive scraping command to execute")
    
    # Source status command
    source_status_parser = adaptive_subparsers.add_parser("status", help="Show source status and scraping statistics")
    source_status_parser.add_argument("--id", type=int, help="Specific source ID to check")
    
    # Retrain source command
    retrain_parser = adaptive_subparsers.add_parser("retrain", help="Retrain selectors for a specific source")
    retrain_parser.add_argument("--id", type=int, required=True, help="Source ID to retrain")
    
    # Source health command
    health_parser = adaptive_subparsers.add_parser("health", help="Analyze source health and get recommendations")
    
    # Update robots.txt cache
    robots_parser = adaptive_subparsers.add_parser("robots", help="Update robots.txt cache for all sources")
    
    # Train ML models
    train_parser = adaptive_subparsers.add_parser("train", help="Train machine learning models for adaptive scraping")
    
    args = parser.parse_args()
    
    aggregator = ContentAggregator(args.config, args.db)
    await aggregator._setup_database()
    aggregator.session = aiohttp.ClientSession()
    
    try:
        if args.command == "update":
            summary = await aggregator.update_content()
            print(f"Update summary: {json.dumps(summary, indent=2)}")
        
        elif args.command == "add":
            categories = args.categories.split(",") if args.categories else []
            success = await aggregator.add_source(args.name, args.url, args.type, categories=categories)
            if success:
                print(f"Source '{args.name}' added successfully.")
                await aggregator.save_config(args.config)
            else:
                print("Failed to add source.")
        
        elif args.command == "remove":
            success = await aggregator.remove_source(args.id)
            if success:
                print(f"Source ID {args.id} removed successfully.")
                await aggregator.save_config(args.config)
            else:
                print(f"Failed to remove source ID {args.id}.")
        
        elif args.command == "list":
            categories = args.categories.split(",") if args.categories else None
            keywords = args.keywords.split(",") if args.keywords else None
            
            articles = await aggregator.get_content(
                limit=args.limit,
                offset=args.offset,
                categories=categories,
                keywords=keywords,
                source_id=args.source,
                sentiment=args.sentiment,
                min_sentiment_score=args.min_sentiment,
                max_sentiment_score=args.max_sentiment,
                include_duplicates=args.include_duplicates,
                similar_to_id=args.similar_to
            )
            
            if not articles:
                print("No articles found matching your criteria.")
            else:
                for article in articles:
                    print(f"\n{'=' * 60}")
                    print(f"ID: {article['id']}")
                    print(f"TITLE: {article['title']}")
                    print(f"SOURCE: {article['source_name']}")
                    print(f"DATE: {article['published']}")
                    
                    if args.similar_to:
                        if 'similarity' in article:
                            print(f"SIMILARITY: {article['similarity']:.2f}")
                    
                    if article.get('is_duplicate'):
                        print("STATUS: Marked as duplicate")
                    
                    if article.get("sentiment_label"):
                        emoji = {
                            'positive': '😀',
                            'neutral': '😐',
                            'negative': '😞'
                        }.get(article["sentiment_label"], '')
                        
                        score = f" ({article['sentiment_score']:.2f})" if article.get("sentiment_score") is not None else ""
                        print(f"SENTIMENT: {article['sentiment_label'].capitalize()} {emoji}{score}")
                    
                    print(f"URL: {article['url']}")
                    content_preview = article['content'][:200].replace('\n', ' ')
                    print(f"\n{content_preview}...")
        
        elif args.command == "export":
            categories = args.categories.split(",") if args.categories else None
            keywords = args.keywords.split(",") if args.keywords else None
            
            success = await aggregator.export_content(
                args.output,
                format=args.format,
                limit=args.limit,
                categories=categories,
                keywords=keywords,
                sentiment=args.sentiment,
                include_duplicates=args.include_duplicates
            )
            
            if success:
                print(f"Content exported to {args.output} in {args.format} format.")
            else:
                print("Export failed.")
        
        elif args.command == "nlp":
            if args.nlp_command == "sentiment":
                stats = await aggregator.get_sentiment_stats()
                print("\n=== SENTIMENT ANALYSIS STATISTICS ===")
                
                if stats.get("counts"):
                    print("SENTIMENT DISTRIBUTION:")
                    emojis = {'positive': '😀', 'neutral': '😐', 'negative': '😞'}
                    for label, count in stats["counts"].items():
                        emoji = emojis.get(label, '')
                        print(f"  {label.capitalize()} {emoji}: {count} articles")
                
                if stats.get("average_score") is not None:
                    print(f"AVERAGE SENTIMENT SCORE: {stats['average_score']:.2f}")
                
                if stats.get("most_positive"):
                    print(f"MOST POSITIVE ARTICLE (Score: {stats['most_positive']['score']:.2f}):")
                    print(f"  ID: {stats['most_positive']['id']}")
                    print(f"  Title: {stats['most_positive']['title']}")
                
                if stats.get("most_negative"):
                    print(f"MOST NEGATIVE ARTICLE (Score: {stats['most_negative']['score']:.2f}):")
                    print(f"  ID: {stats['most_negative']['id']}")
                    print(f"  Title: {stats['most_negative']['title']}")
            
            elif args.nlp_command == "topics":
                top_topics = await aggregator.get_top_topics(args.limit)
                print("\n=== TOP TOPICS ===")
                for i, (topic, count) in enumerate(top_topics, 1):
                    print(f"{i}. {topic} ({count} occurrences)")
            
            elif args.nlp_command == "summarize":
                if args.id:
                    article = await aggregator.get_article_by_id(args.id)
                    if not article:
                        print(f"Article ID {args.id} not found.")
                    else:
                        print(f"\n{'=' * 60}")
                        print(f"TITLE: {article['title']}")
                        print(f"SOURCE: {article['source_name']}")
                        print(f"DATE: {article['published']}")
                        
                        if article.get("summary"):
                            print(f"\n--- SUMMARY ---\n")
                            print(article["summary"])
                        else:
                            print("\nNo summary available. Processing summary now...")
                            summary = await aggregator._generate_summary(article["content"])
                            print(f"\n--- GENERATED SUMMARY ---\n")
                            print(summary)
                        
                        print(f"\n--- FULL CONTENT ---\n")
                        print(article["content"])
                else:
                    articles = await aggregator.get_content(limit=args.limit)
                    for article in articles:
                        print(f"\n{'=' * 60}")
                        print(f"ID: {article['id']}")
                        print(f"TITLE: {article['title']}")
                        print(f"SOURCE: {article['source_name']}")
                        
                        if article.get("summary"):
                            print(f"\n--- SUMMARY ---\n")
                            print(article["summary"])
                        else:
                            print("No summary available.")
            
            elif args.nlp_command == "process":
                stats = await aggregator.process_nlp_for_existing_content(limit=args.limit, force=args.force)
                print("\n=== NLP PROCESSING RESULTS ===")
                print(f"Total articles to process: {stats['total']}")
                print(f"Successfully processed: {stats['processed']}")
                print(f"Errors encountered: {stats['errors']}")
        
        elif args.command == "dedup":
            if args.dedup_command == "run":
                results = await aggregator.deduplicate_existing_content(force=args.force)
                print("\n=== DEDUPLICATION RESULTS ===")
                print(f"Status: {results['status']}")
                
                if "exact_duplicates_found" in results:
                    print(f"Exact duplicates detected: {results['exact_duplicates_found']}")
                
                if "similarity_groups_created" in results:
                    print(f"Similarity groups created: {results['similarity_groups_created']}")
                    print(f"Similar articles detected: {results['similar_articles_found']}")
                
                if "total_processed" in results:
                    print(f"Total articles processed: {results['total_processed']}")
            
            elif args.dedup_command == "stats":
                stats = await aggregator.get_duplicate_stats()
                print("\n=== CONTENT DEDUPLICATION STATISTICS ===")
                print(f"Exact duplicates detected: {stats.get('exact_duplicates', 0)}")
                print(f"Similarity groups: {stats.get('similarity_groups', 0)}")
                print(f"Articles in similarity groups: {stats.get('articles_in_similarity_groups', 0)}")
                
                if stats.get("largest_group"):
                    lg = stats["largest_group"]
                    print(f"Largest similarity group:")
                    print(f"  ID: {lg['id']}")
                    print(f"  Articles: {lg['article_count']}")
                    print(f"  Main article: {lg['main_title']}")
                    print(f"  Source: {lg['main_source']}")
                
                if stats.get("space_saved"):
                    ss = stats["space_saved"]
                    if ss["mb"] > 1:
                        print(f"Estimated storage saved:\n  {ss['mb']:.2f} MB")
                    else:
                        print(f"Estimated storage saved:\n  {ss['kb']:.2f} KB")
                
                if stats.get("recent_duplicates"):
                    print("Recently detected duplicates:")
                    for dup in stats["recent_duplicates"]:
                        print(f"  {dup['title']} (from {dup['source']})")
            
            elif args.dedup_command == "similar":
                articles = await aggregator.get_content(similar_to_id=args.id)
                if not articles:
                    print(f"No similar articles found for ID {args.id}")
                    return
                
                main_article = articles[0]
                similar_articles = articles[1:]
                
                print(f"\n=== MAIN ARTICLE ===")
                print(f"ID: {main_article['id']}")
                print(f"TITLE: {main_article['title']}")
                print(f"SOURCE: {main_article['source_name']}")
                print(f"DATE: {main_article['published']}")
                
                if similar_articles:
                    print(f"\n=== SIMILAR ARTICLES ({len(similar_articles)}) ===")
                    for article in similar_articles:
                        print(f"{'-' * 40}")
                        print(f"ID: {article['id']}")
                        print(f"TITLE: {article['title']}")
                        print(f"SOURCE: {article['source_name']}")
                        print(f"DATE: {article['published']}")
                        
                        if article.get('is_duplicate'):
                            print("STATUS: Marked as duplicate")
                else:
                    print("\nNo similar articles found.")
        
        elif args.command == "adaptive":
            if args.adaptive_command == "status":
                sources = await aggregator.get_source_status(args.id)
                if not sources:
                    print("No source information available.")
                else:
                    for source in sources:
                        print(f"\n{'=' * 60}")
                        print(f"SOURCE: {source['name']} (ID: {source['id']})")
                        print(f"URL: {source['url']}")
                        print(f"TYPE: {source['type']}")
                        print(f"HEALTH: {source['health']}")
                        success_rate = source['success_rate'] * 100 if source['success_rate'] is not None else 0
                        print(f"SUCCESS RATE: {success_rate:.1f}% ({source['access_count']} accesses, {source['error_count']} errors)")
                        print(f"RATE LIMIT: {source['rate_limit']} requests per minute")
                        if source.get('last_access'):
                            print(f"LAST ACCESS: {source['last_access']}")
                        if source.get('top_selectors'):
                            print("\nTOP SELECTORS:")
                            for selector in source['top_selectors']:
                                print(f"  {selector['target']}: {selector['selector']} (score: {selector['score']:.2f})")
            
            elif args.adaptive_command == "retrain":
                result = await aggregator.retrain_source(args.id)
                if result.get('status') == 'success':
                    print(result.get('message'))
                else:
                    print(f"Error: {result.get('message')}")
            
            elif args.adaptive_command == "health":
                health = await aggregator.analyze_source_health()
                if not health or 'status' in health and health['status'] == 'error':
                    print(f"Error analyzing source health: {health.get('message', 'Unknown error')}")
                else:
                    print(f"\n{'=' * 60}")
                    print(f"SOURCE HEALTH SUMMARY")
                    print(f"{'=' * 60}")
                    print(f"Total sources: {health['total_sources']}")
                    print(f"Healthy sources: {health['healthy_sources']} ({health['health_percentage']:.1f}%)")
                    
                    if health.get('recommendations'):
                        print(f"\n{'=' * 60}")
                        print(f"RECOMMENDATIONS")
                        print(f"{'=' * 60}")
                        for rec in health['recommendations']:
                            print(f"- {rec['message']}")
                    
                    if health.get('sources'):
                        print(f"\n{'=' * 60}")
                        print(f"SOURCE DETAILS")
                        print(f"{'=' * 60}")
                        sorted_sources = sorted(health['sources'], key=lambda s: s['health_status'])
                        for source in sorted_sources:
                            status_emoji = "❌" if source['health_status'] == "poor" else "⚠️" if source['health_status'] == "fair" else "✅"
                            print(f"{status_emoji} {source['name']} - {source['health_status'].upper()} (Success rate: {source['success_rate']*100:.1f}%)")
            
            elif args.adaptive_command == "robots":
                result = await aggregator.update_robots_txt_cache()
                if result.get('status') == 'success':
                    print(f"Updated robots.txt cache for {result['updated']} sources.")
                    print(f"Failed: {result['failed']} sources.")
                    print(f"Disallowed by robots.txt: {result['disallowed']} sources.")
                else:
                    print(f"Error updating robots.txt cache: {result.get('message')}")
            
            elif args.adaptive_command == "train":
                print("Training machine learning models for adaptive scraping...")
                await aggregator._train_extraction_models()
                print("Training complete.")
        
        else:
            parser.print_help()
    
    finally:
        await aggregator.close()

if __name__ == "__main__":
    asyncio.run(main())