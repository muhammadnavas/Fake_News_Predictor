import os
import requests
from dotenv import load_dotenv
import time
from typing import List, Dict, Tuple, Optional
from api_utils import is_valid_api_key, should_suppress_error


load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")
CURRENTS_KEY = os.getenv("CURRENTS_KEY")
CONTEXTUALWEB_KEY = os.getenv("ContextualWeb_KEY")


def fetch_newsapi(keyword: Optional[str] = None, country: str = "us", page_size: int = 20) -> List[Dict]:
    """Fetch news from NewsAPI"""
    if not NEWSAPI_KEY:
        print("Warning: NEWSAPI_KEY not found")
        return []
    
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWSAPI_KEY, 
        "country": country, 
        "pageSize": page_size
    }
    
    if keyword:

        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWSAPI_KEY,
            "q": keyword,
            "language": "en",
            "sortBy": "popularity",
            "pageSize": page_size
        }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        articles = []
        for article in response.json().get("articles", []):
            if article.get("title") and article.get("title") != "[Removed]":
                articles.append({
                    "title": article["title"],
                    "description": article.get("description", ""),
                    "url": article["url"],
                    "source": article["source"]["name"],
                    "content": article.get("content", ""),
                    "publishedAt": article.get("publishedAt", ""),
                    "urlToImage": article.get("urlToImage", "")
                })
        
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from NewsAPI: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error with NewsAPI: {e}")
        return []


def fetch_contextualweb(keyword: Optional[str] = None, max_articles: int = 20) -> List[Dict]:
    """Fetch news from ContextualWeb News API"""
    if not CONTEXTUALWEB_KEY:
        print("Warning: ContextualWeb_KEY not found")
        return []
    
    url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"
    params = {
        "q": keyword or "latest",
        "pageNumber": 1,
        "pageSize": max_articles,
        "autoCorrect": "true",
        "safeSearch": "true"
    }
    headers = {
        "x-rapidapi-host": "contextualwebsearch-websearch-v1.p.rapidapi.com",
        "x-rapidapi-key": CONTEXTUALWEB_KEY
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        articles = []
        for article in response.json().get("value", []):
            articles.append({
                "title": article["title"],
                "description": article.get("description", ""),
                "url": article["url"],
                "source": article.get("provider", {}).get("name", "ContextualWeb"),
                "content": article.get("body", ""),
                "publishedAt": article.get("datePublished", ""),
                "urlToImage": article.get("image", {}).get("url", "")
            })
        
        return articles
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from ContextualWeb: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error with ContextualWeb: {e}")
        return []


def fetch_gnews(keyword: Optional[str] = None, max_articles: int = 20) -> List[Dict]:
    """Fetch news from GNews"""
    if not GNEWS_KEY:
        print("Warning: GNEWS_KEY not found")
        return []
    
    url = "https://gnews.io/api/v4/search"
    params = {
        "token": GNEWS_KEY,
        "q": keyword or "latest",
        "max": max_articles,
        "lang": "en"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        articles = []
        for article in response.json().get("articles", []):
            articles.append({
                "title": article["title"],
                "description": article.get("description", ""),
                "url": article["url"],
                "source": article["source"]["name"],
                "content": article.get("content", ""),
                "publishedAt": article.get("publishedAt", ""),
                "urlToImage": article.get("image", "")
            })
        
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from GNews: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error with GNews: {e}")
        return []


def fetch_currents(keyword: Optional[str] = None, page_size: int = 20) -> List[Dict]:
    if not CURRENTS_KEY:
        print("Warning: CURRENTS_KEY not found")
        return []
    

    params = {
        "apiKey": CURRENTS_KEY,
        "language": "en",

    }
    if keyword:
        params["keywords"] = keyword
    
    try:

        response.raise_for_status()
        
        return [
            {
                "title": a["title"],
                "description": a.get("description", ""),
                "url": a["url"],
                "source": a.get("author", "CurrentsAPI"),
                "content": a.get("description", ""),
                "publishedAt": a.get("published", ""),
                "urlToImage": a.get("image", "")
            }
            for a in response.json().get("news", [])
        ]
    except requests.exceptions.Timeout:
        print("⚠️ CurrentsAPI request timed out, skipping...")
        return []
    except Exception as e:
        print(f"Unexpected error with CurrentsAPI: {e}")
        return []


def check_news_existence(news_text: str, country: str = "us") -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if a news article exists by searching for its title or keywords.
    Returns (exists, matched_title, matched_url)
    """
    if not NEWSAPI_KEY:
        print("Warning: NEWSAPI_KEY not found for news existence check")
        return False, None, None
    

    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWSAPI_KEY,

        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        articles = response.json().get("articles", [])
        
        if articles:

            best_match = articles[0]
            return True, best_match["title"], best_match["url"]
        

        words = news_text.lower().split()

        
        if key_words:
            params["q"] = " ".join(key_words)
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            articles = response.json().get("articles", [])
            if articles:
                best_match = articles[0]
                return True, best_match["title"], best_match["url"]
        
        return False, None, None
        
    except requests.exceptions.RequestException as e:
        print(f"Error checking news existence: {e}")
        return False, None, None
    except Exception as e:
        print(f"Unexpected error checking news existence: {e}")
        return False, None, None


def check_newsapi_existence(news_text: str) -> Tuple[bool, List[Dict]]:
    """Enhanced NewsAPI existence check"""
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    if not NEWSAPI_KEY:
        return False, []
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWSAPI_KEY,
            "q": f'"{news_text[:100]}"',
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 10
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        
        if not articles:
            words = news_text.lower().split()
            keywords = [word for word in words if len(word) > 4][:5]
            if keywords:
                params["q"] = " AND ".join(keywords)
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                articles = response.json().get("articles", [])
        
        formatted_articles = []
        for article in articles[:5]:
            if article.get("title") and "[Removed]" not in article["title"]:
                formatted_articles.append({
                    "title": article["title"],
                    "source": article["source"]["name"],
                    "url": article["url"],
                    "publishedAt": article.get("publishedAt", ""),
                    "description": article.get("description", "")
                })
        
        return len(formatted_articles) > 0, formatted_articles
        
    except requests.exceptions.HTTPError as e:
        if not should_suppress_error(e.response.status_code):
            print(f"NewsAPI check failed: {e}")
        return False, []
    except Exception:
        return False, []


def check_gnews_existence(news_text: str) -> Tuple[bool, List[Dict]]:
    """GNews existence check"""
    if not GNEWS_KEY:
        return False, []
    
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "token": GNEWS_KEY,
            "q": news_text[:100],
            "max": 10,
            "lang": "en"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        
        formatted_articles = []
        for article in articles[:5]:
            formatted_articles.append({
                "title": article["title"],
                "source": article["source"]["name"],
                "url": article["url"],
                "publishedAt": article.get("publishedAt", ""),
                "description": article.get("description", "")
            })
        
        return len(formatted_articles) > 0, formatted_articles
        
    except requests.exceptions.HTTPError as e:
        if not should_suppress_error(e.response.status_code):
            print(f"GNews check failed: {e}")
        return False, []
    except Exception:
        return False, []
    
def check_contextualweb_existence(news_text: str, max_results: int = 10) -> Tuple[bool, List[Dict]]:
    """Check if news exists in ContextualWeb"""
    if not CONTEXTUALWEB_KEY:
        return False, []
    
    url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"

        "pageNumber": 1,
        "pageSize": max_results,
        "autoCorrect": "true",
        "safeSearch": "true"
    }
    headers = {
        "x-rapidapi-host": "contextualwebsearch-websearch-v1.p.rapidapi.com",
        "x-rapidapi-key": CONTEXTUALWEB_KEY
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("value", [])
        
        formatted_articles = []
        for article in articles[:5]:
            formatted_articles.append({
                "title": article["title"],
                "source": article.get("provider", {}).get("name", "ContextualWeb"),
                "url": article["url"],
                "publishedAt": article.get("datePublished", ""),
                "description": article.get("description", "")
            })
        
        return len(formatted_articles) > 0, formatted_articles
    
    except requests.exceptions.HTTPError as e:
        if not should_suppress_error(e.response.status_code):
            print(f"ContextualWeb check failed: {e}")
        return False, []
    except Exception:
        return False, []



def fetch_google_fact_checks(query="news", language="en", max_results=10):
    """
    Fetch fact-check claims from Google Fact Check API
    """
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    params = {
        "key": api_key,
        "query": query,
        "languageCode": language,
        "pageSize": max_results
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        claims = []

        for item in data.get("claims", []):
            claims.append({
                "id": item.get("claimant", "unknown") + "_" + item.get("claimDate", ""),
                "text": item.get("text", ""),
                "source": item.get("claimant", "unknown")
            })

        return claims
    except requests.exceptions.HTTPError as e:
        if not should_suppress_error(e.response.status_code):
            print(f"❌ Could not fetch Google Fact Check claims: {e}")
        return []
    except Exception:
        return []


def check_currents_existence(news_text: str) -> Tuple[bool, List[Dict]]:
    """CurrentsAPI existence check"""
    if not CURRENTS_KEY:
        return False, []
    
    try:
        url = "https://api.currentsapi.services/v1/search"
        params = {
            "apiKey": CURRENTS_KEY,
            "keywords": news_text[:100],
            "language": "en",
            "page_size": 10
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("news", [])
        
        formatted_articles = []
        for article in articles[:5]:
            formatted_articles.append({
                "title": article["title"],
                "source": article.get("author", "CurrentsAPI"),
                "url": article["url"],
                "publishedAt": article.get("published", ""),
                "description": article.get("description", "")
            })
        
        return len(formatted_articles) > 0, formatted_articles
        
    except requests.exceptions.HTTPError as e:
        if not should_suppress_error(e.response.status_code):
            print(f"CurrentsAPI check failed: {e}")
        return False, []
    except Exception:
        return False, []


def remove_duplicates(articles: List[Dict]) -> List[Dict]:
    """Remove duplicate articles based on URL and title similarity"""
    seen_urls = set()
    seen_titles = set()
    unique_articles = []
    
    for article in articles:
        url = article.get("url", "")
        title = article.get("title", "").lower().strip()
        
    
        if url in seen_urls or title in seen_titles:
            continue
        

        if not title or len(title) < 10 or title in ["[removed]", "removed"]:
            continue
        
        seen_urls.add(url)
        seen_titles.add(title)
        unique_articles.append(article)
    
    return unique_articles


def get_all_news(keyword: Optional[str] = None, max_articles: int = 50) -> List[Dict]:
    """
    Fetch news from multiple sources and combine them
    Returns deduplicated list of articles
    """
    print(f"Fetching news for keyword: {keyword}")
    all_articles = []
    

    sources = [
        ("NewsAPI", lambda: fetch_newsapi(keyword)),
        ("GNews", lambda: fetch_gnews(keyword, max_articles//3)),
        ("CurrentsAPI", lambda: fetch_currents(keyword, max_articles//3)),
        ("ContextualWeb", lambda: fetch_contextualweb(keyword, max_articles//4))
    ]
    
    for source_name, fetch_func in sources:
        try:
            print(f"Fetching from {source_name}...")
            articles = fetch_func()
            all_articles.extend(articles)
            print(f"Got {len(articles)} articles from {source_name}")

        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            continue
    

    unique_articles = remove_duplicates(all_articles)
    print(f"Total unique articles: {len(unique_articles)}")
    
    return unique_articles[:max_articles]


def search_specific_claim(claim: str, max_results: int = 10) -> List[Dict]:
    """
    Search for specific claims or statements in news articles
    Useful for fact-checking
    """
    if not NEWSAPI_KEY:
        return []
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWSAPI_KEY,

        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        articles = []
        for article in response.json().get("articles", []):
            articles.append({
                "title": article["title"],
                "description": article.get("description", ""),
                "url": article["url"],
                "source": article["source"]["name"],
                "publishedAt": article.get("publishedAt", ""),

            })
        
        return articles
        
    except Exception as e:
        print(f"Error searching for specific claim: {e}")
        return []


def comprehensive_news_check(news_text: str) -> Dict:
    """Check news existence across all available APIs"""
    results = {
        "sources_found": [],
        "total_matches": 0,
        "confidence_score": 0.0,
        "matched_articles": [],
        "search_summary": {}
    }
    
    apis_to_check = [
        ("NewsAPI", lambda: check_newsapi_existence(news_text)),
        ("GNews", lambda: check_gnews_existence(news_text)),
        ("CurrentsAPI", lambda: check_currents_existence(news_text)),

    ]
    
    for api_name, check_func in apis_to_check:
        try:
            exists, articles = check_func()
            results["search_summary"][api_name] = {
                "found": exists,
                "count": len(articles) if articles else 0
            }
            
            if exists and articles:
                results["sources_found"].append(api_name)
                results["total_matches"] += len(articles)
                results["matched_articles"].extend(articles[:3])
        except Exception as e:
            results["search_summary"][api_name] = {
                "found": False,
                "error": str(e)
            }


    google_claims = fetch_google_fact_checks(news_text)
    if google_claims:
        results["sources_found"].append("GoogleFactCheck")
        results["total_matches"] += len(google_claims)
        results["matched_articles"].extend(google_claims[:3])
        results["search_summary"]["GoogleFactCheck"] = {
            "found": True,
            "count": len(google_claims)
        }
    else:
        results["search_summary"]["GoogleFactCheck"] = {"found": False, "count": 0}


    source_count = len(results["sources_found"])
    match_count = min(results["total_matches"], 10)
    results["confidence_score"] = min(
        (source_count * 0.4 + match_count * 0.1) * 100, 100
    )
    
    return results
