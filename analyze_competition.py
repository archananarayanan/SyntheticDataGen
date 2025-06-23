import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    ContentRelevanceFilter
)
from datetime import datetime


async def advanced_crawler():
    filter_chain = FilterChain([
        ContentRelevanceFilter(
            query="What is the latest product announcement",
            threshold=0.8
        )
    ])
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=1,
            include_external=False,
            filter_chain=filter_chain
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
        markdown_generator=DefaultMarkdownGenerator(
            options={"citations": True}
        )
    )
    for source in ["https://news.silabs.com/press-releases", "https://www.nordicsemi.com/Nordic-news"]:
        async with AsyncWebCrawler() as crawler:
            temp = await crawler.arun(source, config=config)
            for result in temp:
                print(result)
                title = result.metadata["title"]
                title = title.replace("/", "_")
                title = title.replace(" ", "_")
                print(title)
                file_name = f"resources/{title}-{datetime.now()}.md"
                try:
                    with open(file_name, "w", encoding="utf-8") as f:
                        f.write(result.markdown)
                except FileExistsError:
                    print(f"File {file_name} already exists")
                    file_name = file_name.replace(".md", "_1.md")
                    with open(file_name, "w", encoding="utf-8") as f:
                        f.write(result.markdown)
                except FileNotFoundError:
                    print(f"File {file_name} not found")
                    with open(file_name, "x", encoding="utf-8") as f:
                        f.write(result.markdown)

if __name__ == "__main__":
    asyncio.run(advanced_crawler())