# from playwright.async_api import async_playwright
# import asyncio


# async def run():
#     async with async_playwright() as p:
#         browser = await p.chromium.launch_persistent_context(headless=False, executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", user_data_dir="C:\\Users\\Soike\\AppData\\Local\\Google\\Chrome\\User Data")
#         page = await browser.new_page()
#         # Start waiting for the download
#         async with page.expect_download() as download_info:
#             # Perform the action that initiates download
#             await page.goto("https://pubs.acs.org/doi/suppl/10.1021/cg200571y/suppl_file/cg200571y_si_001.pdf")
#         download = await download_info.value

#         # Wait for the download process to complete and save the downloaded file somewhere
#         await download.save_as(download.suggested_filename)
        
#         await page.close()
#         # page = await browser.new_page()
#         # await page.goto("https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fange.202312461&file=ange202312461-s1-Supplementary_Information.pdf", wait_until='domcontentloaded')

#         await page.wait_for_timeout(10000)
        

# asyncio.run(run())

import asyncio
from playwright.async_api import async_playwright

download_event = asyncio.Event()

async def handle_download(download):
    # download_bytes = (await download.path()).read_bytes()
    # print("url:", download.url)
    # print("suggested_filename:", download.suggested_filename)
    # print("bytes:")
    # print(download_bytes[:200])
    await download.save_as(download.suggested_filename)
    print(download.url)
    download_event.set()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(headless=False, executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", user_data_dir="C:\\Users\\Soike\\AppData\\Local\\Google\\Chrome\\User Data")
        page = await browser.new_page()
        page.on("download", handle_download)
        try:
            response = await page.goto(
                "https://defret.in/assets/certificates/attestation_secnumacademie.pdf"
            )
        except Exception:
            print("Caught expected exception, waiting on download")
            await download_event.wait()
        else:
            print(await page.title())
            print(await page.content())
            print(await response.body())
        await browser.close()

asyncio.run(main())
