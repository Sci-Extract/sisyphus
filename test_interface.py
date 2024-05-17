from sisyphus.document.document import *
from test_chain import ExtractUptake


user_params = UserParams(False, True, "single-component adsorption isotherms, uptake value are 3 mmol g-1 at 298 K 1 bar", ExtractUptake)
chain = create_chain_by_params(user_params)

document = Document('10.1021&sl;acs.iecr.7b01420.html', user_params)

async def pipeline():
    await document.aindexing()
    await document.aextract_with_chain(chain)

if __name__ == '__main__':
    asyncio.run(pipeline())
