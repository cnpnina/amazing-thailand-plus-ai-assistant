import asyncio
import os
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

async def crawl_page(url: str):

    browser_config = BrowserConfig(verbose=False)
    async with AsyncWebCrawler(config=browser_config) as crawler:

        result = await crawler.arun(
            url=url,
            word_count_threshold=50,
            bypass_cache=True,
            exclude_social_media_links=True,
            remove_overlay_elements=True,
        )

        return {
            "url": url,
            "markdown": result.markdown,
            "title": result.metadata.get("title"),
        }


async def crawl_pages(urls, max_pages=5, max_markdown_chars=12000):
    pages = []
    limited_urls = urls[:max_pages]

    browser_config = BrowserConfig(verbose=False)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in limited_urls:
            try:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=50,
                    bypass_cache=True,
                    exclude_social_media_links=True,
                    remove_overlay_elements=True,
                )
                pages.append(
                    {
                        "url": url,
                        "title": result.metadata.get("title"),
                        "markdown": (result.markdown or "")[:max_markdown_chars],
                    }
                )
            except Exception as exc:
                pages.append(
                    {
                        "url": url,
                        "title": None,
                        "markdown": "",
                        "error": str(exc),
                    }
                )

    return pages


def crawl_pages_sync(urls, max_pages=5, max_markdown_chars=12000):
    return asyncio.run(
        crawl_pages(
            urls=urls,
            max_pages=max_pages,
            max_markdown_chars=max_markdown_chars,
        )
    )


async def main():

    data = await crawl_pages(["https://www.finearts.go.th/"], max_pages=1)
    first = data[0]
    print(first["title"])
    print(first["markdown"][:1000])


if __name__ == "__main__":
    asyncio.run(main())