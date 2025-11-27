import os
import sys
from sisyphus.chain.database import ExtractManager
from sisyphus.chain.constants import *

arg = sys.argv[1]
manager = ExtractManager(
        namespace=arg,
        db_url='sqlite:///' + os.path.join(RECORD_LOCATION, RECORD_NAME),
    )

manager.delete_namespace()